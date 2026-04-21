import os
import shutil
import time
import yaml
import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import get_dataset
from .model import get_model
from .model.tools import count_parameters, safe_model_state_dict
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .utils import (
    use_seed,
    coerce_to_path_and_check_exist,
    coerce_to_path_and_create_dir,
)
from .utils.image import convert_to_img, save_gif
from .utils.logger import get_logger, print_info, print_warning
from .utils.metrics import AverageTensorMeter, AverageMeter, Metrics, Scores
from .utils.path import CONFIGS_PATH, RUNS_PATH
from .utils.plot import plot_lines, plot_bar

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


PRINT_TRAIN_STAT_FMT = "Epoch [{}/{}], Iter [{}/{}], train_metrics: {}".format
PRINT_VAL_STAT_FMT = "Epoch [{}/{}], Iter [{}/{}], val_metrics: {}".format
PRINT_CHECK_CLUSTERS_FMT = (
    "Epoch [{}/{}], Iter [{}/{}]: Reassigned clusters {} from cluster {}".format
)
PRINT_LR_UPD_FMT = "Epoch [{}/{}], Iter [{}/{}], LR update: lr = {}".format

TRAIN_METRICS_FILE = "train_metrics.tsv"
VAL_METRICS_FILE = "val_metrics.tsv"
VAL_SCORES_FILE = "val_scores.tsv"
FINAL_SCORES_FILE = "final_scores.tsv"
MODEL_FILE = "model.pkl"

N_TRANSFORMATION_PREDICTIONS = 4
N_CLUSTER_SAMPLES = 5
MAX_GIF_SIZE = 64
VIZ_HEIGHT = 300
VIZ_WIDTH = 500
VIZ_MAX_IMG_SIZE = 64


class Trainer:
    """Pipeline to train a NN model using a certain dataset, both specified by an YML config."""

    @use_seed()
    def __init__(
        self, cfg, run_dir, save=False
    ):
        self.run_dir = coerce_to_path_and_create_dir(run_dir)
        self.logger = get_logger(self.run_dir, name="trainer")
        self.save_img = save
        self.print_and_log_info(
            "Trainer initialisation: run directory is {}".format(run_dir)
        )

        self.save_img = save
        OmegaConf.save(cfg, self.run_dir/"config.yaml")
        self.print_and_log_info("Current config copied to run directory")

        if torch.cuda.is_available():
            type_device = "cuda"
            nb_device = torch.cuda.device_count()
        else:
            type_device = "cpu"
            nb_device = None
        self.device = torch.device(type_device)
        self.print_and_log_info(
            "Using {} device, nb_device is {}".format(type_device, nb_device)
        )

        # Datasets and dataloaders
        self.dataset_kwargs = cfg["dataset"]
        self.dataset_name = self.dataset_kwargs["name"]
        train_dataset = get_dataset(self.dataset_name)("train", **self.dataset_kwargs)
        val_dataset = get_dataset(self.dataset_name)("val", **self.dataset_kwargs)

        self.n_classes = train_dataset.n_classes
        self.is_val_empty = len(val_dataset) == 0
        self.print_and_log_info(
            "Dataset {} instantiated with {}".format(
                self.dataset_name, self.dataset_kwargs
            )
        )
        self.print_and_log_info(
            "Found {} classes, {} train samples, {} val samples".format(
                self.n_classes, len(train_dataset), len(val_dataset)
            )
        )

        self.img_size = train_dataset.img_size
        self.batch_size = (
            cfg["training"]["batch_size"]
            if cfg["training"]["batch_size"] < len(train_dataset)
            else len(train_dataset)
        )
        self.n_workers = cfg["training"].get("n_workers", 4)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, num_workers=self.n_workers
        )
        self.print_and_log_info(
            "Dataloaders instantiated with batch_size={} and n_workers={}".format(
                self.batch_size, self.n_workers
            )
        )

        self.n_batches = len(self.train_loader)
        self.n_iterations, self.n_epoches = cfg["training"].get("n_iterations"), cfg[
            "training"
        ].get("n_epoches")
        assert not (self.n_iterations is not None and self.n_epoches is not None)
        if self.n_iterations is not None:
            self.n_epoches = max(self.n_iterations // self.n_batches, 1)
        else:
            self.n_iterations = self.n_epoches * len(self.train_loader)

        # Model
        self.model_kwargs = cfg["model"]
        self.model_name = self.model_kwargs["name"]
        self.model = get_model(self.model_name)(
            self.train_loader.dataset, **self.model_kwargs
        ).to(self.device)
        self.print_and_log_info(
            "Using model {} with kwargs {}".format(self.model_name, self.model_kwargs)
        )
        self.print_and_log_info(
            "Number of trainable parameters: {}".format(
                f"{count_parameters(self.model):,}"
            )
        )
        self.n_prototypes = self.model.n_prototypes

        # Optimizer
        opt_params = cfg["training"]["optimizer"] or {}
        optimizer_name = cfg["training"]["optimizer_name"]
        cluster_kwargs = cfg["training"].get("cluster_optimizer", {})
        tsf_kwargs = cfg["training"]["transformer_optimizer"] or {}
        
        self.optimizer = get_optimizer(optimizer_name)(
            [dict(params=self.model.cluster_parameters(), **cluster_kwargs)]
            + [dict(params=self.model.transformer_parameters(), **tsf_kwargs)],
            **opt_params,
        )
        self.model.set_optimizer(self.optimizer)
        self.print_and_log_info(
            "Using optimizer {} with kwargs {}".format(optimizer_name, opt_params)
        )
        self.print_and_log_info("cluster kwargs {}".format(cluster_kwargs))
        self.print_and_log_info("transformer kwargs {}".format(tsf_kwargs))

        # Scheduler
        scheduler_params = cfg["training"].get("scheduler", {})
        scheduler_name = cfg["training"].get("scheduler_name", {}) 
        self.scheduler_update_range = scheduler_params.get("update_range", "epoch")
        assert self.scheduler_update_range in ["epoch", "batch"]
        if scheduler_name == "multi_step" and isinstance(
            scheduler_params["milestones"][0], float
        ):
            n_tot = (
                self.n_epoches
                if self.scheduler_update_range == "epoch"
                else self.n_iterations
            )
            scheduler_params["milestones"] = [
                round(m * n_tot) for m in scheduler_params["milestones"]
            ]
        self.scheduler = get_scheduler(scheduler_name)(
            self.optimizer, **scheduler_params
        )
        self.cur_lr = self.scheduler.get_last_lr()[0]
        self.print_and_log_info(
            "Using scheduler {} with parameters {}".format(
                scheduler_name, scheduler_params
            )
        )

        # Pretrained / Resume
        checkpoint_path = cfg["training"].get("pretrained")
        checkpoint_path_resume = cfg["training"].get("resume")
        assert not (checkpoint_path is not None and checkpoint_path_resume is not None)
        if checkpoint_path is not None:
            self.load_from_tag(checkpoint_path)
        elif checkpoint_path_resume is not None:
            self.load_from_tag(checkpoint_path_resume, resume=True)
        else:
            self.start_epoch, self.start_batch = 1, 1

        # Train metrics & check_cluster interval
        metric_names = ["time/img", "loss"]
        metric_names += [f"prop_clus{i}" for i in range(self.n_prototypes)]
        metric_names += [f"proba_clus{i}" for i in range(self.n_prototypes)]
        self.bin_edges = np.arange(0, 1.1, 0.1)
        self.bin_counts = np.zeros(len(self.bin_edges) - 1)
        train_iter_interval = cfg["training"]["train_stat_interval"]
        self.train_stat_interval = train_iter_interval
        self.train_metrics = Metrics(*metric_names)
        self.train_metrics_path = self.run_dir / TRAIN_METRICS_FILE
        with open(self.train_metrics_path, mode="w") as f:
            f.write(
                "iteration\tepoch\tbatch\t" + "\t".join(self.train_metrics.names) + "\n"
            )
        self.check_cluster_interval = cfg["training"]["check_cluster_interval"]

        # Val metrics & scores
        val_iter_interval = cfg["training"]["val_stat_interval"]
        self.val_stat_interval = val_iter_interval
        val_metric_names = ["loss_val"]
        train_iter_interval = cfg["training"]["train_stat_interval"]
        self.val_metrics = Metrics(*val_metric_names)
        self.val_metrics_path = self.run_dir / VAL_METRICS_FILE
        with open(self.val_metrics_path, mode="w") as f:
            f.write(
                "iteration\tepoch\tbatch\t" + "\t".join(self.val_metrics.names) + "\n"
            )

        self.val_scores = Scores(self.n_classes, self.n_prototypes)
        self.val_scores_path = self.run_dir / VAL_SCORES_FILE
        with open(self.val_scores_path, mode="w") as f:
            f.write(
                "iteration\tepoch\tbatch\t" + "\t".join(self.val_scores.names) + "\n"
            )

        # Prototypes & Variances
        if self.save_img:
            self.prototypes_path = coerce_to_path_and_create_dir(
                self.run_dir / "prototypes"
            )
            [
                coerce_to_path_and_create_dir(self.prototypes_path / f"proto{k}")
                for k in range(self.n_prototypes)
            ]

            # Transformation predictions
            self.transformation_path = coerce_to_path_and_create_dir(
                self.run_dir / "transformations"
            )
            self.images_to_tsf = next(iter(self.train_loader))[0][
                :N_TRANSFORMATION_PREDICTIONS
            ].to(self.device)
            for k in range(self.images_to_tsf.size(0)):
                out = coerce_to_path_and_create_dir(self.transformation_path / f"img{k}")
                convert_to_img(self.images_to_tsf[k]).save(out / "input.png")
                [
                    coerce_to_path_and_create_dir(out / f"tsf{k}")
                    for k in range(self.n_prototypes)
                ]

    def print_and_log_info(self, string):
        print_info(string)
        self.logger.info(string)

    def load_from_tag(self, tag, resume=False):
        self.print_and_log_info("Loading model from run {}".format(tag))
        path = coerce_to_path_and_check_exist(
            RUNS_PATH / self.dataset_name / tag / MODEL_FILE
        )
        checkpoint = torch.load(path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            state = safe_model_state_dict(checkpoint["model_state"])
            self.model.load_state_dict(state)
        self.start_epoch, self.start_batch = 1, 1
        if resume:
            self.start_epoch, self.start_batch = (
                checkpoint["epoch"],
                checkpoint.get("batch", 0) + 1,
            )
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            if hasattr(self, "sprite_optimizer"):
                self.sprite_optimizer.load_state_dict(
                    checkpoint["sprite_optimizer_state"]
                )
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.cur_lr = self.scheduler.get_last_lr()[0]
        self.print_and_log_info(
            "Checkpoint loaded at epoch {}, batch {}".format(
                self.start_epoch, self.start_batch - 1
            )
        )

    @property
    def score_name(self):
        return self.val_scores.score_name

    def print_memory_usage(self, prefix):
        usage = {}
        for attr in [
            "memory_allocated",
            "max_memory_allocated",
            "memory_cached",
            "max_memory_cached",
        ]:
            usage[attr] = getattr(torch.cuda, attr)() * 0.000001
        self.print_and_log_info(
            "{}:\t{}".format(
                prefix,
                " / ".join(["{}: {:.0f}MiB".format(k, v) for k, v in usage.items()]),
            )
        )

    @use_seed()
    def run(self):
        cur_iter = (self.start_epoch - 1) * self.n_batches + self.start_batch - 1
        prev_train_stat_iter, prev_val_stat_iter = cur_iter, cur_iter
        prev_check_cluster_iter = cur_iter
        if self.start_epoch == self.n_epoches + 1:
            self.print_and_log_info("No training, only evaluating")

            self.evaluate()
            self.print_and_log_info("Training run is over")
            return

        for epoch in range(self.start_epoch, self.n_epoches + 1):
            batch_start = self.start_batch if epoch == self.start_epoch else 1
            for batch, (images, labels, _, _) in enumerate(self.train_loader, start=1):
                if batch < batch_start:
                    continue
                cur_iter += 1
                if cur_iter > self.n_iterations:
                    break

                self.single_train_batch_run(images, labels)
                if self.scheduler_update_range == "batch":
                    self.update_scheduler(epoch, batch=batch)

                if (cur_iter - prev_train_stat_iter) >= self.train_stat_interval:
                    prev_train_stat_iter = cur_iter
                    self.log_train_metrics(cur_iter, epoch, batch)

                if (cur_iter - prev_check_cluster_iter) >= self.check_cluster_interval:
                    prev_check_cluster_iter = cur_iter
                    self.check_cluster(cur_iter, epoch, batch)

                if (cur_iter - prev_val_stat_iter) >= self.val_stat_interval:
                    prev_val_stat_iter = cur_iter
                    if not self.is_val_empty:
                        self.run_val()
                        self.log_val_metrics(cur_iter, epoch, batch)
                    if self.save_img:
                        self.log_images(cur_iter)
                    self.save(epoch=epoch, batch=batch)

            self.model.step()
            if self.scheduler_update_range == "epoch" and batch_start == 1:
                self.update_scheduler(epoch + 1, batch=1)

        self.save_training_metrics()
        self.evaluate()

        self.print_and_log_info("Training run is over")

    def update_scheduler(self, epoch, batch):
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        if lr != self.cur_lr:
            self.cur_lr = lr
            self.print_and_log_info(
                PRINT_LR_UPD_FMT(epoch, self.n_epoches, batch, self.n_batches, lr)
            )

    def single_train_batch_run(self, images, labels):
        start_time = time.time()
        B = images.size(0)
        self.model.train()
        images = images.to(self.device)

        self.optimizer.zero_grad()
        loss, distances, probas = self.model(images)
        loss.backward()
        self.optimizer.step()

        if hasattr(self, "sprite_optimizer"):
            self.sprite_optimizer.step()

        with torch.no_grad():
            if hasattr(self.model, "proba"):
                argmin_idx = probas.argmax(1)  # probas: B, K
            else:
                argmin_idx = distances.min(1)[1]
            mask = torch.zeros(B, self.n_prototypes, device=self.device).scatter(
                1, argmin_idx[:, None], 1
            )

            if hasattr(self.model, "proba"):
                winners = probas * mask  # B, K
                probabilities = (
                    winners.sum(0).cpu().numpy() / mask.sum(0).cpu().numpy()
                )  # K
                isnan = np.isnan(probabilities)
                probabilities[isnan] = 0
                self.train_metrics.update(
                    {f"proba_clus{i}": p for i, p in enumerate(probabilities)}
                )
            proportions = mask.sum(0).cpu().numpy() / B
            argmin_idx = argmin_idx.cpu().numpy()

            self.train_metrics.update(
                {
                    "time/img": (time.time() - start_time) / B,
                    "loss": loss.item(),
                }
            )
            self.train_metrics.update(
                {f"prop_clus{i}": p for i, p in enumerate(proportions)}
            )

    @torch.no_grad()
    def log_images(self, cur_iter):
        self.save_prototypes(cur_iter)
        tsf_imgs = self.save_transformed_images(cur_iter)
        C, H, W = tsf_imgs.shape[2:]

    @torch.no_grad()
    def save_prototypes(self, cur_iter=None):
        prototypes = self.model.prototypes
        for k in range(self.n_prototypes):
            img = convert_to_img(prototypes[k])
            if cur_iter is not None:
                img.save(self.prototypes_path / f"proto{k}" / f"{cur_iter}.jpg")
            else:
                img.save(self.prototypes_path / f"prototype{k}.png")

    @torch.no_grad()
    def save_transformed_images(self, cur_iter=None):
        self.model.eval()
        output = self.model.transform(self.images_to_tsf)

        transformed_imgs = torch.cat([self.images_to_tsf.unsqueeze(1), output], 1)
        for k in range(transformed_imgs.size(0)):
            for j, img in enumerate(transformed_imgs[k][1:]):
                if cur_iter is not None:
                    convert_to_img(img).save(
                        self.transformation_path
                        / f"img{k}"
                        / f"tsf{j}"
                        / f"{cur_iter}.jpg"
                    )
                else:
                    convert_to_img(img).save(
                        self.transformation_path / f"img{k}" / f"tsf{j}.png"
                    )
        return transformed_imgs

    def check_cluster(self, cur_iter, epoch, batch):
        proportions = [
            self.train_metrics[f"prop_clus{i}"].avg for i in range(self.n_prototypes)
        ]
        reassigned, idx = self.model.reassign_empty_clusters(proportions)
        msg = PRINT_CHECK_CLUSTERS_FMT(
            epoch, self.n_epoches, batch, self.n_batches, reassigned, idx
        )
        self.print_and_log_info(msg)
        self.train_metrics.reset(*[f"prop_clus{i}" for i in range(self.n_prototypes)])
        self.train_metrics.reset(*[f"proba_clus{i}" for i in range(self.n_prototypes)])

    def log_train_metrics(self, cur_iter, epoch, batch):
        # Print & write metrics to file
        stat = PRINT_TRAIN_STAT_FMT(
            epoch, self.n_epoches, batch, self.n_batches, self.train_metrics
        )
        self.print_and_log_info(stat)
        with open(self.train_metrics_path, mode="a") as f:
            f.write(
                "{}\t{}\t{}\t".format(cur_iter, epoch, batch)
                + "\t".join(map("{:.4f}".format, self.train_metrics.avg_values))
                + "\n"
            )

        self.train_metrics.reset("time/img", "loss")

    @torch.no_grad()
    def run_val(self):
        self.model.eval()
        for images, labels, _, _ in self.val_loader:
            images = images.to(self.device)
            if hasattr(self.model, "proba"):
                _, out, probas = self.model(images)
                argmin_idx = probas.argmax(1)
                dist_min_by_sample = out[:, argmin_idx]
            else:
                distances = self.model(images)[1]
                dist_min_by_sample, argmin_idx = distances.min(1)
            loss_val = dist_min_by_sample.mean()

            self.val_metrics.update({"loss_val": loss_val.item()})
            self.val_scores.update(labels.long().numpy(), argmin_idx.cpu().numpy())

    def log_val_metrics(self, cur_iter, epoch, batch):
        stat = PRINT_VAL_STAT_FMT(
            epoch, self.n_epoches, batch, self.n_batches, self.val_metrics
        )
        self.print_and_log_info(stat)
        with open(self.val_metrics_path, mode="a") as f:
            f.write(
                "{}\t{}\t{}\t".format(cur_iter, epoch, batch)
                + "\t".join(map("{:.4f}".format, self.val_metrics.avg_values))
                + "\n"
            )

        scores = self.val_scores.compute()
        self.print_and_log_info(
            "val_scores: "
            + ", ".join(["{}={:.4f}".format(k, v) for k, v in scores.items()])
        )
        with open(self.val_scores_path, mode="a") as f:
            f.write(
                "{}\t{}\t{}\t".format(cur_iter, epoch, batch)
                + "\t".join(map("{:.4f}".format, scores.values()))
                + "\n"
            )

        self.val_scores.reset()
        self.val_metrics.reset()

    def save(self, epoch, batch):
        state = {
            "epoch": epoch,
            "batch": batch,
            "model_name": self.model_name,
            "model_kwargs": self.model_kwargs,
            "model_state": self.model.state_dict(),
            "n_prototypes": self.n_prototypes,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        if hasattr(self, "sprite_optimizer"):
            state["sprite_optimizer_state"] = self.sprite_optimizer.state_dict()
        save_path = self.run_dir / MODEL_FILE
        torch.save(state, save_path)
        self.print_and_log_info("Model saved at {}".format(save_path))

    def save_training_metrics(self):
        df_train = pd.read_csv(self.train_metrics_path, sep="\t", index_col=0)
        df_val = pd.read_csv(self.val_metrics_path, sep="\t", index_col=0)
        df_scores = pd.read_csv(self.val_scores_path, sep="\t", index_col=0)
        if len(df_train) == 0:
            self.print_and_log_info("No metrics or plots to save")
            return

        # Losses
        losses = list(filter(lambda s: s.startswith("loss"), self.train_metrics.names))
        # df = df_train.join(df_val[["loss_val"]], how="outer")
        df = df_train
        fig = plot_lines(df, losses, title="Loss")  # + ["loss_val"], title="Loss")
        fig.savefig(self.run_dir / "loss.pdf")

        # Cluster proportions
        names = list(filter(lambda s: s.startswith("prop_"), self.train_metrics.names))
        fig = plot_lines(df, names, title="Cluster proportions")
        fig.savefig(self.run_dir / "cluster_proportions.pdf")
        s = df[names].iloc[-1]
        s.index = list(map(lambda n: n.replace("prop_clus", ""), names))
        fig = plot_bar(s, title="Final cluster proportions")
        fig.savefig(self.run_dir / "cluster_proportions_final.pdf")

        # Cluster probabilities
        names = list(filter(lambda s: s.startswith("proba_"), self.train_metrics.names))
        fig = plot_lines(df, names, title="Cluster Probabilities")
        fig.savefig(self.run_dir / "cluster_probabilities.pdf")

        # Validation
        if not self.is_val_empty:
            names = list(filter(lambda name: "cls" not in name, self.val_scores.names))
            fig = plot_lines(df_scores, names, title="Global scores", unit_yaxis=True)
            fig.savefig(self.run_dir / "global_scores.pdf")

            fig = plot_lines(
                df_scores,
                [f"acc_cls{i}" for i in range(self.n_classes)],
                title="Scores by cls",
                unit_yaxis=True,
            )
            fig.savefig(self.run_dir / "scores_by_cls.pdf")

        # Prototypes & Variances
        if self.save_img:
            size = MAX_GIF_SIZE if MAX_GIF_SIZE < max(self.img_size) else self.img_size
            with torch.no_grad():
                self.save_prototypes()
            for k in range(self.n_prototypes):
                save_gif(
                    self.prototypes_path / f"proto{k}", f"prototype{k}.gif", size=size
                )
                shutil.rmtree(str(self.prototypes_path / f"proto{k}"))

            # Transformation predictions
            if self.model.transformer.is_identity:
                # no need to keep transformation predictions
                shutil.rmtree(str(self.transformation_path))
                coerce_to_path_and_create_dir(self.transformation_path)
            else:
                self.save_transformed_images()
                for i in range(self.images_to_tsf.size(0)):
                    for k in range(self.n_prototypes):
                        save_gif(
                            self.transformation_path / f"img{i}" / f"tsf{k}",
                            f"tsf{k}.gif",
                            size=size,
                        )
                        shutil.rmtree(
                            str(self.transformation_path / f"img{i}" / f"tsf{k}")
                        )

        self.print_and_log_info("Training metrics and visuals saved")

    def evaluate(self):
        self.model.eval()
        no_label = self.train_loader.dataset[0][1] == -1
        if no_label:
            self.qualitative_eval()
        else:
            # self.qualitative_eval()
            self.quantitative_eval()
        self.print_and_log_info("Evaluation is over")

    @torch.no_grad()
    def qualitative_eval(self):
        """Routine to save qualitative results"""
        loss = AverageMeter()
        scores_path = self.run_dir / FINAL_SCORES_FILE
        with open(scores_path, mode="w") as f:
            f.write("loss\n")

        cluster_path = coerce_to_path_and_create_dir(self.run_dir / "clusters")
        dataset = self.train_loader.dataset
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            shuffle=False,
        )
        # Compute results
        distances, cluster_idx = np.array([]), np.array([], dtype=np.int32)
        averages = {k: AverageTensorMeter() for k in range(self.n_prototypes)}
        cluster_by_path = []
        for images, _, _, path in train_loader:
            images = images.to(self.device)

            if hasattr(self.model, "proba"):
                _, out, probas = self.model(images)
                argmin_idx = probas.argmax(1)
                argmin_idx = argmin_idx.cpu().numpy()
                dist_min_by_sample = out[:, argmin_idx].cpu().numpy()
            else:
                out = self.model(images)[1]
                dist_min_by_sample, argmin_idx = map(
                    lambda t: t.cpu().numpy(), out.min(1)
                )

            loss.update(dist_min_by_sample.mean(), n=len(dist_min_by_sample))
            argmin_idx = argmin_idx.astype(np.int32)
            distances = np.hstack([distances, dist_min_by_sample])
            cluster_idx = np.hstack([cluster_idx, argmin_idx])
            if hasattr(train_loader.dataset, "data_path"):
                cluster_by_path += [
                    (os.path.relpath(p, train_loader.dataset.data_path), argmin_idx[i])
                    for i, p in enumerate(path)
                ]
            transformed_imgs = self.model.transform(images).cpu()
            for k in range(self.n_prototypes):
                imgs = transformed_imgs[argmin_idx == k, k]
                averages[k].update(imgs)
        if cluster_by_path:
            cluster_by_path = pd.DataFrame(
                cluster_by_path, columns=["path", "cluster_id"]
            ).set_index("path")
            cluster_by_path.to_csv(self.run_dir / "cluster_by_path.csv")

        self.print_and_log_info("final_loss: {:.5}".format(loss.avg))

        # Save results
        with open(cluster_path / "cluster_counts.tsv", mode="w") as f:
            f.write("\t".join([str(k) for k in range(self.n_prototypes)]) + "\n")
            f.write(
                "\t".join([str(averages[k].count) for k in range(self.n_prototypes)])
                + "\n"
            )
        for k in range(self.n_prototypes):
            path = coerce_to_path_and_create_dir(cluster_path / f"cluster{k}")
            indices = np.where(cluster_idx == k)[0]
            top_idx = np.argsort(distances[indices])[:N_CLUSTER_SAMPLES]
            for j, idx in enumerate(top_idx):
                inp = dataset[indices[idx]][0].unsqueeze(0).to(self.device)
                convert_to_img(inp).save(path / f"top{j}_raw.png")
                if not self.model.transformer.is_identity:
                    convert_to_img(self.model.transform(inp)[0, k]).save(
                        path / f"top{j}_tsf.png"
                    )
            if len(indices) <= N_CLUSTER_SAMPLES:
                random_idx = indices
            else:
                random_idx = np.random.choice(indices, N_CLUSTER_SAMPLES, replace=False)
            for j, idx in enumerate(random_idx):
                inp = dataset[idx][0].unsqueeze(0).to(self.device)
                convert_to_img(inp).save(path / f"random{j}_raw.png")
                if not self.model.transformer.is_identity:
                    convert_to_img(self.model.transform(inp)[0, k]).save(
                        path / f"random{j}_tsf.png"
                    )
            try:
                convert_to_img(averages[k].avg).save(path / "avg.png")
            except AssertionError:
                print_warning(f"no image found in cluster {k}")

    @torch.no_grad()
    def quantitative_eval(self):
        """Routine to save quantitative results: loss + scores"""
        loss = AverageMeter()
        scores_path = self.run_dir / FINAL_SCORES_FILE
        scores = Scores(self.n_classes, self.n_prototypes)
        with open(scores_path, mode="w") as f:
            f.write("loss\t" + "\t".join(scores.names) + "\n")

        dataset = get_dataset(self.dataset_name)(
            "train", eval_mode=True, **self.dataset_kwargs
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, num_workers=self.n_workers
        )
        for images, labels, _, _ in loader:
            images = images.to(self.device)
            if hasattr(self.model, "proba"):
                _, out, probas = self.model(images)
                argmin_idx = probas.argmax(1)
                dist_min_by_sample = out[:, argmin_idx]
                hist, _ = np.histogram(probas.cpu().numpy(), bins=self.bin_edges)
                self.bin_counts += hist
            else:
                distances = self.model(images)[1]
                dist_min_by_sample, argmin_idx = distances.min(1)

            loss.update(dist_min_by_sample.mean(), n=len(dist_min_by_sample))
            scores.update(labels.long().numpy(), argmin_idx.cpu().numpy())

        scores = scores.compute()
        self.print_and_log_info("bin_counts: " + str(self.bin_counts))
        self.print_and_log_info("final_loss: {:.5}".format(loss.avg))
        self.print_and_log_info(
            "final_scores: "
            + ", ".join(["{}={:.4f}".format(k, v) for k, v in scores.items()])
        )
        with open(scores_path, mode="a") as f:
            f.write(
                "{:.5}\t".format(loss.avg)
                + "\t".join(map("{:.4f}".format, scores.values()))
                + "\n"
            )


import datetime
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dataset = cfg["dataset"]["name"]
    seed = cfg["training"]["seed"]
    save = cfg["training"]["save"]
    day = datetime.datetime.now().day
    month = datetime.datetime.now().month
    job_name = HydraConfig.get().job.name

    hid = HydraConfig.get()
    job_id = hid.job.get("num", 0)

    tag = f"{dataset}_{job_name}_{job_id}"

    if cfg["training"]["cont"] == True:
        cfg["training"]["resume"] = tag

    run_dir = RUNS_PATH / dataset / tag
    run_dir = str(run_dir)
    trainer = Trainer(cfg, run_dir, seed=seed, save=save)
    try:
        trainer.run(seed=seed)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == "__main__":
    main()
