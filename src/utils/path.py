from pathlib import Path
from environ import Env

# Project and source files
DEFAULT_PROJECT_PATH = Path(__file__).parent.parent.parent

# Allow overriding the project path with an environment variable
ENV = Env()
PROJECT_PATH = Path(ENV("DTI_DATA_FOLDER", default=DEFAULT_PROJECT_PATH))

DATASET_SCRATCH = PROJECT_PATH 

CONFIGS_PATH = PROJECT_PATH / 'configs'
DATASETS_PATH = DATASET_SCRATCH / 'datasets'
RUNS_PATH = DATASET_SCRATCH / 'runs'
RESULTS_PATH = PROJECT_PATH / 'results' 
