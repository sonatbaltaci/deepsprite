from copy import deepcopy
from itertools import chain
import math

import torch
from torch.optim import Adam, RMSprop
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torchvision
import torch.nn.functional as F

from .transformer import (
    PrototypeTransformationNetwork as Transformer,
    N_HIDDEN_UNITS,
    N_LAYERS,
)
from .tools import (
    copy_with_noise,
    generate_data,
    create_gaussian_weights,
    get_clamp_func,
    create_mlp,
    get_bbox_from_mask,
)
from ..utils.logger import print_warning
from .u_net import UNet

import numpy as np

NOISE_SCALE = 0.0001
EMPTY_CLUSTER_THRESHOLD = 0.2
LATENT_SIZE = 128

def softmax(logits, tau=1., dim=-1):
    return F.softmax(logits/tau, dim=dim)

def init_linear(hidden, out, init, n_channels=3, std=5, value=0.9, dataset=None, uniform=[None, None]):
    if init == "random":
        return nn.Linear(hidden, out)
    elif init == "constant":
        linear = nn.Linear(hidden, out)
        h = int(math.sqrt(out / n_channels))
        nn.init.constant_(linear.weight, 1e-10)
        sample = torch.full(size=(n_channels * h * h,), fill_value=value)
        sample = torch.log(sample / (1 - sample))
        linear.bias.data.copy_(sample)
        return linear
    elif init == "gaussian":
        assert n_channels == 1
        print_warning("Last layer initialized with gaussian weights.")
        linear = nn.Linear(hidden, out)
        h = int(math.sqrt(out))
        size = [h, h]
        sample = create_gaussian_weights(size, 1, std).flatten()
        nn.init.constant_(
            linear.weight, 1e-10
        )  # a small value to avoid vanishing grads
        sample = torch.log(sample / (1 - sample))
        linear.bias.data.copy_(sample)
        return linear
    elif init == "mean":
        linear = nn.Linear(hidden, out)
        assert dataset is not None
        images = next(
            iter(DataLoader(dataset, batch_size=100, shuffle=True, num_workers=4))
        )[0]
        sample = images.mean(0)
        nn.init.constant_(linear.weight, 0.0001)
        sample = torch.log(sample / (1 - sample))
        linear.bias.data.copy_(sample.flatten())
        return linear
    else:
        raise NotImplementedError("init is not implemented.")


def layered_composition(layers, masks, occ_grid, proba=False):
    # LBCHW size of layers and masks and LLB size for occ_grid
    if proba:
        occ_masks = (1 - occ_grid[..., None, None, None].transpose(0, 1) * masks.sum(dim=1)).prod(
            1
        )  # LBCHW
        final_layer = (masks*layers).sum(1) # LBCHW
        return (occ_masks * final_layer).sum(0)  # BCHW
    else:
        occ_masks = (1 - occ_grid[..., None, None, None].transpose(0, 1) * masks).prod(1)  # LBCHW
        return (occ_masks * masks * layers).sum(0)  # BCHW


def softmax(logits, tau=1., dim=-1):
    return F.softmax(logits/tau, dim=dim)


class MultiLayer(nn.Module):
    name = "multilayer"
    learn_masks = True

    def __init__(self, n_epochs, dataset, n_sprites, n_objects=1, **kwargs):
        super().__init__()
        if dataset is None:
            raise NotImplementedError
        else:
            img_size = dataset.img_size
            n_ch = dataset.n_channels

        self.count = 0

        self.num_epochs = n_epochs
        # Prototypes & masks
        size = kwargs.get("sprite_size", img_size)
        self.sprite_size = size
        self.img_size = img_size
        self.return_map_out = kwargs.get("return_map_out", False)
        color_channels = kwargs.get("color_channels", 3)
        self.color_channels = color_channels
        self.add_empty_sprite = kwargs.get("add_empty_sprite", False)
        self.lambda_empty_sprite = kwargs.get("lambda_empty_sprite", 0)
        self.n_sprites = n_sprites + 1 if self.add_empty_sprite else n_sprites

        proto_args = kwargs.get("prototype")
        proto_source = proto_args.get("source", "data")
        assert proto_source in ["data", "generator"]
        self.proto_source = proto_source
        if proto_args.get("data", None) is not None:
            data_args = proto_args.get("data")
            freeze_frg, freeze_bkg, freeze_sprite = data_args.get("freeze", [0, 0, 0])
            value_frg, value_bkg, value_mask = data_args.get("value", [0.5, 0.5, 0.5])
            std = data_args.get("gaussian_weights_std", 25)
            proto_init, bkg_init, mask_init = data_args.get(
                "init", ["constant", "constant", "constant"]
            )
            print(freeze_frg, freeze_bkg, freeze_sprite)
        else:
            freeze_frg, freeze_bkg, freeze_sprite = False, False, False
            value_frg, value_bkg, value_mask = 0.5, 0.5, 0.5
            std = 25
            proto_init, bkg_init, mask_init = "constant", "constant", "constant"
        self.freeze_frg_milestone = freeze_frg if freeze_frg else -1
        freeze_frg = False

        if proto_source == "data":
            if freeze_frg:
                self.prototype_params = nn.Parameter(
                    torch.stack(
                        generate_data(
                            dataset, n_sprites, proto_init, value=value_frg, size=size
                        )
                    ),
                    requires_grad=False,
                )
            else:
                self.prototype_params = nn.Parameter(
                    torch.stack(
                        generate_data(
                            dataset, n_sprites, proto_init, value=value_frg, size=size
                        )
                    )
                )

            self.mask_params = nn.Parameter(
                self.init_masks(n_sprites, mask_init, size, std, value_mask, dataset)
            )
        else:
            if freeze_frg:
                self.prototype_params = nn.Parameter(
                    torch.stack(
                        generate_data(
                            dataset, n_sprites, proto_init, value=value_frg, size=size
                        )
                    ),
                    requires_grad=False,
                )

            gen_name = proto_args.get("generator", "mlp")
            print_warning("Sprites will be generated from latent variables.")
            assert gen_name in ["mlp", "unet"]
            latent_dims = (LATENT_SIZE,) if gen_name == "mlp" else (1, size[0], size[1])
            self.latent_params = nn.Parameter(
                torch.stack(
                    [
                        torch.normal(mean=0.0, std=1.0, size=latent_dims)
                        for k in range(n_sprites)
                    ],
                    dim=0,
                ),
            )
            if not freeze_frg:
                self.frg_generator = self.init_generator(
                    gen_name,
                    LATENT_SIZE,
                    color_channels,
                    color_channels * size[0] * size[1],
                    proto_init,
                    std=std,
                    value=value_frg,
                )
            self.mask_generator = self.init_generator(
                gen_name,
                LATENT_SIZE,
                1,
                1 * size[0] * size[1],
                mask_init,
                std=std,
                value=0.0,
            )
        clamp_name = kwargs.get("use_clamp", "soft")
        self.clamp_func = get_clamp_func(clamp_name)
        self.cur_epoch = 0
        self.n_linear_layers = kwargs.get("n_linear_layers", N_LAYERS)
        self.estimate_minimum = kwargs.get("estimate_minimum", False)
        self.greedy_algo_iter = kwargs.get("greedy_algo_iter", 1)
        self.freeze_milestone = freeze_sprite if freeze_sprite else -1
        assert isinstance(self.freeze_milestone, (int,))
        self.freeze_frg = freeze_frg
        self.freeze_bkg = freeze_bkg
        
        softmax_f = kwargs.get("softmax", "softmax")
        if kwargs.get("learn_tau", False):
            self.learn_tau=True
            t_ = kwargs.get("tau",1)
            self.tau = nn.Parameter(torch.tensor([t_],dtype=torch.float, device="cuda"))
        else:
            self.learn_tau=False
            self.tau = kwargs.get("tau",1)
        
        # Sprite transformers
        L = n_objects
        self.n_objects = n_objects
        self.has_layer_tsf = kwargs.get(
            "transformation_sequence_layer", "identity"
        ) not in ["id", "identity"]
        if self.has_layer_tsf:
            layer_kwargs = deepcopy(kwargs)
            layer_kwargs["transformation_sequence"] = kwargs[
                "transformation_sequence_layer"
            ]
            layer_kwargs["curriculum_learning"] = kwargs["curriculum_learning_layer"]
            self.layer_transformer = Transformer(n_ch, img_size, L, **layer_kwargs)
            self.encoder = self.layer_transformer.encoder
            tsfs = [
                Transformer(
                    n_ch,
                    size,
                    self.n_sprites,
                    layer_size=img_size,
                    encoder=self.encoder,
                    **dict(kwargs, freeze_frg=freeze_frg),
                )
                for k in range(L)
            ]
            self.sprite_transformers = nn.ModuleList(tsfs)
        else:
            if L > 1:
                self.layer_transformer = Transformer(
                    n_ch, img_size, L, transformation_sequence="identity"
                )
            first_tsf = Transformer(
                n_ch, img_size, self.n_sprites, **dict(kwargs, freeze_frg=freeze_frg)
            )
            self.encoder = first_tsf.encoder
            tsfs = [
                Transformer(
                    n_ch,
                    img_size,
                    self.n_sprites,
                    encoder=self.encoder,
                    **dict(kwargs, freeze_frg=freeze_frg),
                )
                for k in range(L - 1)
            ]
            self.sprite_transformers = nn.ModuleList([first_tsf] + tsfs)

        # Background Transformer
        M = kwargs.get("n_backgrounds", 0)
        self.n_backgrounds = M
        self.learn_backgrounds = M > 0
        if self.learn_backgrounds:
            if proto_source == "data":
                self.bkg_params = nn.Parameter(
                    torch.stack(
                        generate_data(dataset, M, init_type=bkg_init, value=value_bkg)
                    )
                )
                self.bkg_params.requires_grad = False if freeze_bkg else True
            else:
                if freeze_bkg:
                    self.bkg_params = nn.Parameter(
                        torch.stack(
                            generate_data(
                                dataset, M, init_type=bkg_init, value=value_bkg
                            )
                        ),
                        requires_grad=False,
                    )
                else:
                    gen_name = proto_args.get("generator", "mlp")
                    print_warning("Background will be generated from latent variables.")
                    latent_dims = (
                        (LATENT_SIZE,)
                        if gen_name == "mlp"
                        else (1, img_size[0], img_size[1])
                    )
                    self.bkg_generator = self.init_generator(
                        gen_name,
                        LATENT_SIZE,
                        color_channels,
                        color_channels * img_size[0] * img_size[1],
                        kwargs.get("init_bkg_linear", "random"),
                        std=None,
                        dataset=dataset,
                        value=value_bkg,
                    )
                    self.latent_bkg_params = nn.Parameter(
                        torch.stack(
                            [
                                torch.normal(mean=0.0, std=1.0, size=latent_dims)
                                for k in range(M)
                            ]
                        )
                    )

            bkg_kwargs = deepcopy(kwargs)
            bkg_kwargs["transformation_sequence"] = kwargs[
                "transformation_sequence_bkg"
            ]
            bkg_kwargs["curriculum_learning"] = kwargs["curriculum_learning_bkg"]
            bkg_kwargs["padding_mode"] = "border"
            self.bkg_transformer = Transformer(
                n_ch, img_size, M, encoder=self.encoder, **bkg_kwargs
            )

        # Image composition and aux
        self.pred_occlusion = kwargs.get("pred_occlusion", False)
        if self.pred_occlusion:
            nb_out = int(L * (L - 1) / 2)
            norm = kwargs.get("norm_layer")
            self.occ_predictor = create_mlp(
                self.encoder.out_ch, nb_out, N_HIDDEN_UNITS, self.n_linear_layers, norm
            )
            self.occ_predictor[-1].weight.data.zero_()
            self.occ_predictor[-1].bias.data.zero_()
        else:
            self.register_buffer("occ_grid", torch.tril(torch.ones(L, L), diagonal=-1))

        self._criterion = nn.MSELoss(reduction="none")
        self.empty_cluster_threshold = kwargs.get(
            "empty_cluster_threshold", EMPTY_CLUSTER_THRESHOLD / n_sprites
        )
        self._reassign_cluster = kwargs.get("reassign_cluster", True)
        self.inject_noise = kwargs.get("inject_noise", 0)

        proba = kwargs.get("proba", False)
        if proba:
            softmax_f = kwargs.get("softmax","gumbel_softmax")
            self.softmax_f = softmax if softmax_f == "softmax" else F.gumbel_softmax
            self.proba_type = kwargs.get("proba_type", "marionette")
            if self.proba_type == "linear":  # linear mapping
                #self.proba = nn.Sequential(nn.Linear(self.encoder.out_ch, self.n_sprites * n_objects), nn.LayerNorm(self.n_sprites * n_objects, elementwise_affine=False))
                self.proba = nn.Linear(self.encoder.out_ch, self.n_sprites * n_objects)
            else:  # marionette-like
                self.proba = [nn.Sequential(
                    nn.Linear(self.encoder.out_ch, LATENT_SIZE),
                    nn.LayerNorm(LATENT_SIZE, elementwise_affine=False),
                ).to("cuda") for _ in range(self.n_objects)]
                self.empty_latent_params = nn.Parameter(
                    torch.normal(mean=0.0, std=1.0, size=latent_dims)
                )
            self.freq_weight = kwargs.get("freq_weight", 0)
            self.bin_weight = kwargs.get("bin_weight", 0)
            self.start_bin_weight = self. bin_weight # 0.0001
            if self.bin_weight <= self.start_bin_weight:
                self.curr_bin_weight = self.bin_weight
            else:
                self.curr_bin_weight = self.start_bin_weight
            self.beta_dist = torch.distributions.Beta(
                torch.Tensor([2.0]).to("cuda"), torch.Tensor([2.0]).to("cuda")
            )
        self.estimate_proba = proba

    @staticmethod
    def init_masks(K, mask_init, size, std, value, dataset):
        if mask_init == "constant":
            masks = torch.full((K, 1, *size), value)
        elif mask_init == "gaussian":
            assert std is not None
            mask = create_gaussian_weights(size, 1, std)
            masks = mask.unsqueeze(0).expand(K, -1, -1, -1).clone()
        elif mask_init == "random":
            masks = torch.rand(K, 1, *size)
        elif mask_init == "sample":
            assert dataset
            masks = torch.stack(
                generate_data(dataset, K, init_type=mask_init, value=value)
            )
            assert masks.shape[1] == 1
        else:
            raise NotImplementedError(f"unknown mask_init: {mask_init}")
        return masks

    @staticmethod
    def init_generator(
        name,
        latent_dim,
        color_channel,
        out_channel,
        init="random",
        value=0.9,
        std=5,
        dataset=None,
    ):
        if name == "unet":
            return UNet(1, color_channel)
        elif name == "mlp":
            linear = init_linear(
                8 * latent_dim,
                out_channel,
                init,
                n_channels=color_channel,
                std=std,
                dataset=dataset,
                value=value,
            )
            model = nn.Sequential(
                nn.Linear(latent_dim, 8 * latent_dim),
                nn.GroupNorm(8, 8 * latent_dim),
                nn.ReLU(inplace=True),
                linear,
                nn.Sigmoid(),
            )
            return model
        else:
            raise NotImplementedError("Generator not implemented.")

    @property
    def n_prototypes(self):
        return self.n_sprites

    @property
    def masks(self):
        if self.proto_source == "data":
            masks = self.mask_params
        else:
            with torch.no_grad():
                masks = self.mask_generator(self.latent_params)
            if len(masks.size()) != 4:
                masks = masks.reshape(-1, 1, self.sprite_size[0], self.sprite_size[1])

        if self.add_empty_sprite:
            masks = torch.cat(
                [masks, torch.zeros(1, *masks[0].shape, device=masks.device)]
            )

        if self.inject_noise and self.training:
            return masks
        else:
            return self.clamp_func(masks)

    @property
    def prototypes(self):
        if self.proto_source == "data":
            params = self.prototype_params
        else:
            with torch.no_grad():
                if self.freeze_frg:
                    params = self.prototype_params
                else:
                    params = self.frg_generator(self.latent_params)
                    if len(params.size()) != 4:
                        params = params.reshape(
                            -1,
                            self.color_channels,
                            self.sprite_size[0],
                            self.sprite_size[1],
                        )

        if self.add_empty_sprite:
            params = torch.cat(
                [params, torch.zeros(1, *params[0].shape, device=params.device)]
            )

        return self.clamp_func(params)

    @property
    def backgrounds(self):
        if self.proto_source == "data":
            params = self.bkg_params
        else:
            with torch.no_grad():
                if self.freeze_bkg:
                    params = self.bkg_params
                else:
                    params = self.bkg_generator(self.latent_bkg_params)
                    if len(params.size()) != 4:
                        params = params.reshape(
                            -1, self.color_channels, self.img_size[0], self.img_size[1]
                        )
        return self.clamp_func(params)

    @property
    def is_layer_tsf_id(self):
        if hasattr(self, "layer_transformer"):
            return self.layer_transformer.only_id_activated
        else:
            return False

    @property
    def are_sprite_frozen(self):
        return (
            True
            if self.freeze_milestone > 0 and self.cur_epoch < self.freeze_milestone
            else False
        )

    @property
    def are_frg_frozen(self):
        return (
            True
            if self.freeze_frg_milestone > 0 and self.cur_epoch < self.freeze_frg_milestone
            else False
        )
    
    def cluster_parameters(self):
        if self.proto_source == "data":
            params = [self.prototype_params, self.mask_params]
            if self.learn_backgrounds:
                params.append(self.bkg_params)
        else:
            params = list(chain(*[self.mask_generator.parameters()])) + [
                self.latent_params
            ]
            if not self.freeze_frg:
                params.extend(list(chain(*[self.frg_generator.parameters()])))
            if self.learn_backgrounds and not self.freeze_bkg:
                params.append(self.latent_bkg_params)
                params.extend(list(chain(*[self.bkg_generator.parameters()])))
        if self.estimate_proba:
            if self.proba_type == "marionette":
                params.append(self.empty_latent_params)
                for p in self.proba:
                    params.extend(list(chain(*[p.parameters()])))
            else:
                if self.learn_tau:
                    params.append(self.tau)
                params.extend(list(chain(*[self.proba.parameters()])))
        return iter(params)

    def transformer_parameters(self):
        params = [t.get_parameters() for t in self.sprite_transformers]
        if hasattr(self, "layer_transformer"):
            params.append(self.layer_transformer.get_parameters())
        if self.learn_backgrounds:
            params.append(self.bkg_transformer.get_parameters())
        if self.pred_occlusion:
            params.append(self.occ_predictor.parameters())
        return chain(*params)

    def reg_func(self, probas, type="freq"):
        if type == "freq":
            probas_ = probas[:, :-1, :] if self.add_empty_sprite else probas
            probas_ = probas_ / torch.max(probas_, dim=1, keepdim=True)[0] # just like reassignment of clusters from proportion
            freqs = probas_.mean(dim=-1).flatten() # mean over B, dim: KxL
            freqs = freqs / freqs.sum()
            return freqs.clamp(max=(self.empty_cluster_threshold))
        elif type == "bin":
            if self.are_sprite_frozen:
                return torch.Tensor([0.0]).to(probas.device)
            p = probas.clamp(min=1e-5, max=1 - 1e-5)  # LKB
            return torch.exp(self.beta_dist.log_prob(p))
        elif type == "empty_sprite":
            r = (self.lambda_empty_sprite * torch.Tensor(
                    [1] * (self.n_sprites - 1) + [0])).to(probas.device).reshape(
                    1, self.n_sprites, 1) # 1K1
            r = (r * probas).mean()
            return r
        else:
            raise ValueError("undefined regularizer")

    def estimate_logits(self, features):
        if self.proba_type == "marionette":
            if self.add_empty_sprite:
                latent_params = torch.cat([self.latent_params, self.empty_latent_params.unsqueeze(0)], dim=0)
            else: 
                latent_params = self.latent_params # KD
            latent_params = torch.nn.functional.layer_norm(latent_params, (latent_params.shape[-1],))
            proba_theta = [self.proba[l](features) for l in range(self.n_objects)]
            proba_theta = torch.stack(proba_theta, dim=2).permute(1,0,2) # DBL
            D, B, L = proba_theta.shape
            temp = torch.matmul(latent_params, proba_theta.reshape(D,-1)).reshape(-1, B, L).permute(1,2,0)
            logits = (1.0 / np.sqrt(self.encoder.out_ch)) * (temp) # BLK
            return logits
        elif self.proba_type == "linear":
            logits = self.proba(features).reshape(features.shape[0], self.n_objects, self.n_sprites)
            return logits

    def forward(self, x, img_masks=None):
        loss_em = torch.Tensor([0.0])
        B, C, H, W = x.size()
        L, K, M = self.n_objects, self.n_sprites, self.n_backgrounds or 1
        tsf_layers, tsf_masks, tsf_bkgs, occ_grid, class_prob = self.predict(
            x
        )
        if class_prob is None:
            target = self.compose(
                tsf_layers, tsf_masks, occ_grid, tsf_bkgs, class_prob
            )  # B(K**L*M)CHW
            x = x.unsqueeze(1).expand(-1, K**L * M, -1, -1, -1)
            if img_masks != None:
                img_masks = img_masks.unsqueeze(1).expand(-1, K**L * M, -1, -1, -1)
            distances = self.criterion(x, target, weights=img_masks)
            loss_r = distances.min(1)[0].mean()
            loss = (loss_r, loss_r, torch.Tensor([0.0]), torch.Tensor([0.0]))
        else:
            target = self.compose(
                tsf_layers, tsf_masks, occ_grid, tsf_bkgs, class_prob
            )  # BCHW
            if img_masks != None:
                img_masks = img_masks.unsqueeze(1)
            if self.estimate_proba:
                freq_loss = self.reg_func(class_prob, type="freq")
                bin_loss = self.reg_func(class_prob, type="bin")

                loss_r = self.criterion(
                    x.unsqueeze(1), target.unsqueeze(1), weights=img_masks
                ).mean()
                if self.add_empty_sprite and not self.are_sprite_frozen:
                    loss_em = self.reg_func(class_prob, type="empty_sprite")
                    loss_r += loss_em
                loss_freq = 1 - freq_loss.sum()
                loss_bin = bin_loss.mean()
                loss_all = (
                    loss_r + self.freq_weight * loss_freq + self.curr_bin_weight * loss_bin
                )
                class_oh = torch.zeros(class_prob.shape, device=x.device).scatter_(
                    1, class_prob.argmax(1, keepdim=True), 1
                )
                distances = 1 - class_oh.permute(2, 0, 1).flatten(1)  # B(L*K)
                loss = (loss_all, loss_r, loss_bin, loss_freq, loss_em)
            else:
                loss_r = self.criterion(
                    x.unsqueeze(1), target.unsqueeze(1), weights=img_masks
                ).mean()
                distances = 1 - class_prob.permute(2, 0, 1).flatten(1)  # B(L*K)
                loss = (loss_r, loss_r, torch.Tensor([0.0]), torch.Tensor([0.0]), loss_em)

        return loss, distances, class_prob

    def predict(self, x):
        B, C, H, W = x.size()
        h, w = self.prototypes.shape[2:]
        L, K, M = self.n_objects, self.n_sprites, self.n_backgrounds or 1
        if hasattr(self, "mask_generator"):
            masks = self.mask_generator(self.latent_params)
            masks = masks.reshape(-1, 1, self.sprite_size[0], self.sprite_size[1])
            if self.freeze_frg:
                prototypes = self.prototypes
            else:
                prototypes = self.frg_generator(self.latent_params).reshape(
                    -1, self.color_channels, self.sprite_size[0], self.sprite_size[1]
                )
            if self.add_empty_sprite:
                if not self.freeze_frg:
                    prototypes = torch.cat(
                        [
                            prototypes,
                            torch.zeros(
                                1, *prototypes[0].shape, device=prototypes.device
                            ),
                        ]
                    )
                masks = torch.cat(
                    [masks, torch.zeros(1, *masks[0].shape, device=masks.device)]
                )
            prototypes = self.clamp_func(prototypes)
            if self.inject_noise and self.training:
                masks = masks
            else:
                masks = self.clamp_func(masks)
        else:
            prototypes = self.prototypes
            masks = self.masks

        prototypes = prototypes.unsqueeze(1).expand(K, B, C, -1, -1)
        if self.are_frg_frozen:
            prototypes = prototypes.detach()
        masks = masks.unsqueeze(1).expand(K, B, 1, -1, -1)
        sprites = torch.cat([prototypes, masks], dim=2)
        if self.inject_noise and self.training:
            # XXX we use a canva to inject noise after transformations to avoid gridding artefacts
            if self.add_empty_sprite:
                canvas = torch.cat(
                    [torch.ones(K - 1, B, 1, h, w), torch.zeros(1, B, 1, h, w)]
                ).to(x.device)
            else:
                canvas = torch.ones(K, B, 1, h, w, device=x.device)
            sprites = torch.cat([sprites, canvas], dim=2)
        if self.are_sprite_frozen:
            sprites = sprites.detach()
        features = self.encoder(x)
        tsf_sprites = torch.stack(
            [self.sprite_transformers[k](x, sprites, features)[1] for k in range(L)],
            dim=0,
        )
        if self.has_layer_tsf:
            layer_features = features.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
            if tsf_sprites.shape[-1] == W:
                h, w = tsf_sprites.shape[-2:]
            tsf_layers = self.layer_transformer(
                x, tsf_sprites.view(L, B * K, -1, h, w), layer_features
            )[1]
            tsf_layers = tsf_layers.view(B, K, L, -1, H, W).transpose(0, 2)  # LKBCHW
        else:
            tsf_layers = tsf_sprites.transpose(1, 2)  # LKBCHW

        if self.inject_noise and self.training:
            tsf_layers, tsf_masks, tsf_noise = torch.split(tsf_layers, [C, 1, 1], dim=3)
        else:
            tsf_layers, tsf_masks = torch.split(tsf_layers, [C, 1], dim=3)

        if self.learn_backgrounds:
            if hasattr(self, "mask_generator"):
                if not self.freeze_bkg:
                    backgrounds = self.bkg_generator(self.latent_bkg_params).reshape(
                        -1, self.color_channels, self.img_size[0], self.img_size[1]
                    )
                    backgrounds = self.clamp_func(backgrounds)
                else:
                    backgrounds = self.backgrounds
            else:
                backgrounds = self.backgrounds
            backgrounds = backgrounds.unsqueeze(1).expand(M, B, C, -1, -1)
            tsf_bkgs = self.bkg_transformer(x, backgrounds, features)[1].transpose(
                0, 1
            )  # MBCHW
        else:
            tsf_bkgs = None

        if self.inject_noise and self.training:  #  and epoch >= 500:
            noise = (
                torch.rand(K, 1, H, W, device=x.device)[None, None, ...]
                .expand(L, B, K, 1, H, W)
                .transpose(1, 2)
            )
            tsf_masks = tsf_masks + tsf_noise * (
                2 * self.inject_noise * noise - self.inject_noise
            )
            tsf_masks = self.clamp_func(tsf_masks)

        occ_grid = self.predict_occlusion_grid(x, features)  # LLB

        if self.estimate_proba:
            logits = self.estimate_logits(features) 
            if self.learn_tau:
                tau = F.softplus(self.tau)
            else:
                tau = self.tau
            if self.add_empty_sprite and self.are_sprite_frozen:
                logits = logits[:, :, :-1] # B, L, K-1
                if self.training:
                    class_prob = self.softmax_f(logits, tau, dim=-1).permute(1, 2, 0) # LKB
                else:
                    class_prob = softmax(logits, tau, dim=-1).permute(1, 2, 0) # LKB
                class_prob = torch.cat([class_prob, torch.zeros(L, 1, B, device=class_prob.device)], dim=1)
            else:
                if self.training:
                    class_prob = self.softmax_f(logits, tau, dim=-1).permute(1, 2, 0) # LKB
                else:
                    class_prob = softmax(logits, tau, dim=-1).permute(1, 2, 0) # LKB
        else:
            if self.estimate_minimum:
                class_prob = self.greedy_algo_selection(
                    x, tsf_layers, tsf_masks, tsf_bkgs, occ_grid
                )  # LKB
                self._class_prob = class_prob  # for monitoring and debug only
            else:
                class_prob = None

        return tsf_layers, tsf_masks, tsf_bkgs, occ_grid, class_prob

    def predict_occlusion_grid(self, x, features):
        B, L = x.size(0), self.n_objects
        if self.pred_occlusion:
            inp = features if features is not None else x
            occ_grid = self.occ_predictor(inp)  # view(-1, L, L)
            occ_grid = torch.sigmoid(occ_grid)
            grid = torch.zeros(B, L, L, device=x.device)
            indices = torch.tril_indices(row=L, col=L, offset=-1)
            grid[:, indices[0], indices[1]] = occ_grid
            occ_grid = grid + torch.triu(1 - grid.transpose(1, 2), diagonal=1)
        else:
            occ_grid = self.occ_grid.unsqueeze(0).expand(B, -1, -1)

        return occ_grid.permute(1, 2, 0)  # LLB

    @torch.no_grad()
    def greedy_algo_selection(self, x, layers, masks, bkgs, occ_grid):
        L, K, B, C, H, W = layers.shape
        if self.add_empty_sprite and self.are_sprite_frozen: # 1. exclude empty sprite if frozen
            layers, masks = layers[:, :-1], masks[:, :-1]
            K = K - 1
        x, device = x.unsqueeze(0).expand(K, -1, -1, -1, -1), x.device
        bkgs = torch.zeros(1, B, C, H, W, device=device) if bkgs is None else bkgs
        cur_layers = torch.cat([bkgs, torch.zeros(L, B, C, H, W, device=device)])
        cur_masks = torch.cat(
            [
                torch.ones(1, B, 1, H, W, device=device),
                torch.zeros(L, B, 1, H, W, device=device),
            ]
        )
        one, zero = torch.ones(B, L, 1, device=device), torch.zeros(
            B, 1, L + 1, device=device
        )
        occ_grid = torch.cat(
            [zero, torch.cat([one, occ_grid.permute(2, 0, 1)], dim=2)], dim=1
        ).permute(1, 2, 0)

        resps, diff_select = torch.zeros(L, K, B, device=device), [[], []]
        for step in range(self.greedy_algo_iter):
            for l, (layer, mask) in enumerate(zip(layers, masks), start=1):
                recons = []
                for k in range(K):
                    tmp_layers = torch.cat(
                        [cur_layers[:l], layer[[k]], cur_layers[l + 1 :]]
                    )
                    tmp_masks = torch.cat(
                        [cur_masks[:l], mask[[k]], cur_masks[l + 1 :]]
                    )
                    recons.append(layered_composition(tmp_layers, tmp_masks, occ_grid))
                distance = ((x - torch.stack(recons)) ** 2).flatten(2).mean(2)
                if self.add_empty_sprite and not self.are_sprite_frozen:
                    distance += (
                        self.lambda_empty_sprite
                        * torch.Tensor([1] * (K - 1) + [0]).to(device)[:, None]
                    )

                resp = torch.zeros(K, B, device=device).scatter_(
                    0, distance.argmin(0, keepdim=True), 1
                )
                resps[l - 1] = resp
                cur_layers[l] = (layer * resp[..., None, None, None]).sum(axis=0)
                cur_masks[l] = (mask * resp[..., None, None, None]).sum(axis=0)

            if True:
                # For debug purposes only
                if step == 0:
                    indices = resps.argmax(1).flatten()
                else:
                    new_indices = resps.argmax(1).flatten()
                    diff_select[0].append(str(step))
                    diff_select[1].append(
                        (new_indices != indices).float().mean().item()
                    )
                    indices = new_indices
        # For debug purposes only
        if step > 0:
            self._diff_selections = diff_select

        if self.add_empty_sprite and self.are_sprite_frozen:
            resps = torch.cat([resps, torch.zeros(L, 1, B, device=device)], dim=1)
        return resps

    def compose(self, layers, masks, occ_grid, backgrounds=None, class_prob=None):
        L, K, B, C, H, W = layers.shape
        device = occ_grid.device

        is_binary = (class_prob == 0) | (class_prob == 1)
        is_only_0_or_1 = is_binary.all()

        if class_prob is not None:
            if is_only_0_or_1:
                masks = (masks * class_prob[..., None, None, None]).sum(axis=1)
                layers = (layers * class_prob[..., None, None, None]).sum(axis=1)

                if backgrounds is not None:
                    masks = torch.cat([torch.ones(1, B, 1, H, W, device=device), masks]) # why 1, not M?
                    layers = torch.cat([backgrounds, layers])
                    one, zero = torch.ones(B, L, 1, device=device), torch.zeros(
                        B, 1, L + 1, device=device
                    )
                    occ_grid = torch.cat(
                        [zero, torch.cat([one, occ_grid.permute(2, 0, 1)], dim=2)], dim=1
                    ).permute(1, 2, 0)

                return layered_composition(layers, masks, occ_grid)
            else:
                masks = (masks * class_prob[..., None, None, None]) 
                if backgrounds is not None:
                    masks = torch.cat([torch.ones(1, K, B, 1, H, W, device=device)/K, masks]) # why 1, not M?
                    M, _, _, _, _ = backgrounds.shape
                    backgrounds = backgrounds.unsqueeze(1).expand(-1, K, -1, -1, -1, -1)
                    layers = torch.cat([backgrounds, layers])
                    one, zero = torch.ones(B, L, 1, device=device), torch.zeros(
                        B, 1, L + 1, device=device
                    )
                    occ_grid = torch.cat(
                        [zero, torch.cat([one, occ_grid.permute(2, 0, 1)], dim=2)], dim=1
                    ).permute(1, 2, 0)

                return layered_composition(layers, masks, occ_grid, proba=True)

        else:
            layers = [
                layers[k][(None,) * (L - 1)].transpose(k, L - 1) for k in range(L)
            ]  # L elements of size K1.. 1BCHW
            masks = [
                masks[k][(None,) * (L - 1)].transpose(k, L - 1) for k in range(L)
            ]  # L elements of size K1...1BCHW
            size = (K,) * L + (B, C, H, W)
            if backgrounds is not None:
                M = backgrounds.size(0)
                backgrounds = backgrounds[(None,) * L].transpose(0, L)  # M1..1BCHW
                layers = [backgrounds] + [layers[k][None] for k in range(L)]
                masks = [torch.ones((1,) * (L + 1) + (B, C, H, W)).to(device)] + [
                    masks[k][None] for k in range(L)
                ]
                one, zero = torch.ones(B, L, 1, device=device), torch.zeros(
                    B, 1, L + 1, device=device
                )
                occ_grid = torch.cat(
                    [zero, torch.cat([one, occ_grid.permute(2, 0, 1)], dim=2)], dim=1
                ).permute(1, 2, 0)
                size = (M,) + size
            else:
                M = 1

            occ_grid = occ_grid[..., None, None, None]
            res = torch.zeros(size, device=device)
            for k in range(len(layers)):
                if backgrounds is not None:
                    j_start = 1 if self.pred_occlusion else k + 1
                else:
                    j_start = 0 if self.pred_occlusion else k + 1
                occ_masks = torch.ones(size, device=device)
                for j in range(j_start, len(layers)):
                    if j != k:
                        occ_masks *= 1 - occ_grid[j, k] * masks[j]
                res += occ_masks * masks[k] * layers[k]
            return res.view(K**L * M, B, C, H, W).transpose(0, 1)

    def criterion(self, inp, target, weights=None, reduction="mean"):
        dist = self._criterion(inp, target)
        if weights is not None:
            dist = dist * weights
        if reduction == "mean":
            return dist.flatten(2).mean(2)
        elif reduction == "sum":
            return dist.flatten(2).sum(2)
        elif reduction == "none":
            return dist
        else:
            raise NotImplementedError

    @torch.no_grad()
    def transform(
        self,
        x,
        with_composition=False,
        pred_semantic_labels=False,
        pred_instance_labels=False,
        with_bkg=True,
        hard_occ_grid=False,
    ):
        B, C, H, W = x.size()
        L, K = self.n_objects, self.n_sprites

        tsf_layers, tsf_masks, tsf_bkgs, occ_grid, class_prob = self.predict(x)
        if class_prob is not None:
            class_oh = torch.zeros(class_prob.shape, device=x.device).scatter_(
                1, class_prob.argmax(1, keepdim=True), 1
            )
        else:
            class_oh = None

        if pred_semantic_labels:
            label_layers = (
                torch.arange(1, K + 1, device=x.device)[(None,) * 4]
                .transpose(0, 4)
                .expand(L, -1, B, 1, H, W)
            )
            true_occ_grid = (occ_grid > 0.5).float()
            target = self.compose(
                label_layers,
                (tsf_masks > 0.5).long(),
                true_occ_grid,
                class_prob=class_oh
            ).squeeze(1)
            if self.return_map_out:
                bboxes = get_bbox_from_mask(tsf_masks)
                class_ids = class_oh
                return target.clamp(0, self.n_sprites).long(), bboxes, class_ids
            else:
                return target.clamp(0, self.n_sprites).long()

        elif pred_instance_labels:
            label_layers = (
                torch.arange(1, L + 1, device=x.device)[(None,) * 5]
                .transpose(0, 5)
                .expand(-1, K, B, 1, H, W)
            )
            true_occ_grid = (occ_grid > 0.5).float()
            target = self.compose(
                label_layers,
                (tsf_masks > 0.5).long(),
                true_occ_grid,
                class_prob=class_oh,
            ).squeeze(1)
            target = target.clamp(0, L).long()
            if not with_bkg and class_oh is not None:
                bkg_idx = target == 0
                tsf_layers = (tsf_layers * class_oh[..., None, None, None]).sum(axis=1)
                new_target = ((tsf_layers - x) ** 2).sum(2).argmin(0).long() + 1
                target[bkg_idx] = new_target[bkg_idx]
            return target

        else:
            occ_grid = (occ_grid > 0.5).float() if hard_occ_grid else occ_grid
            tsf_layers, tsf_masks = tsf_layers.clamp(0, 1), tsf_masks.clamp(0, 1)
            
            if tsf_bkgs is not None:
                tsf_bkgs = tsf_bkgs.clamp(0, 1)

            if hard_occ_grid: # Threshold also the class probability for visualization
                class_prob = class_oh
            target = self.compose(tsf_layers, tsf_masks, occ_grid, tsf_bkgs, class_prob)
            if class_prob is not None:
                target = target.unsqueeze(1)

            if with_composition:
                compo = []
                for k in range(L):
                    compo += [
                        tsf_layers[k].transpose(0, 1),
                        tsf_masks[k].transpose(0, 1),
                    ]
                if self.learn_backgrounds:
                    compo.insert(2, tsf_bkgs.transpose(0, 1))
                return target, compo, class_prob
            else:
                return target

    def step(self):
        self.cur_epoch += 1
        [tsf.step() for tsf in self.sprite_transformers]
        if hasattr(self, "layer_transformer"):
            self.layer_transformer.step()
        if self.learn_backgrounds:
            self.bkg_transformer.step()
        if hasattr(self, "proba") and self.curr_bin_weight < self.bin_weight:
            if not self.are_sprite_frozen:
                self.curr_bin_weight = self.start_bin_weight + (self.bin_weight - self.start_bin_weight) * ((self.cur_epoch-self.freeze_milestone) / 40)
                print(f"Updating bin weight to {self.curr_bin_weight}")
        # if hasattr(self, "proba") and self.cur_epoch == 25:
        #    self.curr_bin_weight = 0.01
        #    print(f"Updating bin weight to {self.curr_bin_weight}")


    def set_optimizer(self, opt):
        self.optimizer = opt
        [tsf.set_optimizer(opt) for tsf in self.sprite_transformers]
        if hasattr(self, "layer_transformer"):
            self.layer_transformer.set_optimizer(opt)
        if self.learn_backgrounds:
            self.bkg_transformer.set_optimizer(opt)

    def load_state_dict(self, state_dict):
        unloaded_params = []
        state = self.state_dict()
        for name, param in state_dict.items():
            if name in state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                if "activations" in name and state[name].shape != param.shape:
                    state[name].copy_(
                        torch.Tensor([True] * state[name].size(0)).to(param.device)
                    )
                else:
                    state[name].copy_(param)
            elif name == "prototypes":
                state["prototype_params"].copy_(param)
            elif name == "backgrounds":
                state["bkg_params"].copy_(param)
            else:
                unloaded_params.append(name)
        if len(unloaded_params) > 0:
            print_warning(f"load_state_dict: {unloaded_params} not found")

    def reassign_empty_clusters(self, proportions):
        if not self._reassign_cluster or self.are_sprite_frozen:
            return [], 0
        if self.add_empty_sprite:
            proportions = proportions[:-1] / max(proportions[:-1])

        N, threshold = len(proportions), self.empty_cluster_threshold
        reassigned = []
        idx = torch.argmax(proportions).item()
        for i in range(N):
            if proportions[i] < threshold:
                self.restart_branch_from(i, idx)
                reassigned.append(i)
                # break  # if one cluster is split, stop reassigning
        if len(reassigned) > 0:
            self.restart_branch_from(idx, idx)

        return reassigned, idx

    def restart_branch_from(self, i, j):
        if hasattr(self, "mask_generator"):
            self.latent_params[i].data.copy_(
                copy_with_noise(self.latent_params[j], NOISE_SCALE)
            )
            params = [self.latent_params]
        else:
            self.mask_params[i].data.copy_(self.mask_params[j].detach().clone())
            params = [self.mask_params]
            if not self.freeze_frg:
                self.prototype_params[i].data.copy_(
                    copy_with_noise(self.prototype_params[j], NOISE_SCALE)
                )
                params.extend([self.prototype_params])
        [
            tsf.restart_branch_from(i, j, noise_scale=0)
            for tsf in self.sprite_transformers
        ]

        if hasattr(self, "optimizer"):
            opt = self.optimizer
            if isinstance(opt, (Adam,)):
                for param in params:
                    if "exp_avg" in opt.state[param]:
                        opt.state[param]["exp_avg"][i] = opt.state[param]["exp_avg"][j]
                        opt.state[param]["exp_avg_sq"][i] = opt.state[param]["exp_avg_sq"][
                            j
                        ]
                    else:
                        # make sure we are here because frg is frozen (not in the graph yet)
                        assert self.are_frg_frozen
            elif isinstance(opt, (RMSprop,)):
                for param in params:
                    opt.state[param]["square_avg"][i] = opt.state[param]["square_avg"][
                        j
                    ]
            else:
                raise NotImplementedError(
                    "unknown optimizer: you should define how to reinstanciate statistics if any"
                )
