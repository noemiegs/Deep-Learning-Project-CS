# Chemin de base pour ExperimentRuns
DATA_DIR = "data/raw/train/static/ExperimentRuns"
TARGET_DIR = "data/raw/train/overlay/ExperimentRuns"
SRC_DIR = "src"

USE_AUGMENTATION = False

VOXEL_SIZES = {
    '0': 10,  # Pleine résolution
    '1': 20,  # Résolution intermédiaire
    '2': 40   # Résolution basse
}

PARTICLE_COLORS = {
    "apo-ferritin": "red",
    "beta-amylase": "blue",
    "ribosome": "purple",
    "thyroglobulin": "orange",
    "virus-like-particle": "cyan",
    "beta-galactosidase": "green"
}
DEFAULT_COLOR = "red"

CLASS_MAPPING = {
    "background": 0,
    "apo-ferritin": 1,
    "beta-amylase": 2,
    "ribosome": 3,
    "thyroglobulin": 4,
    "virus-like-particle": 5,
    "beta-galactosidase": 6
}

PARTICLE_WEIGHTS = {
    'background': 1,
    'apo-ferritin': 1,
    'beta-amylase': 1,
    'beta-galactosidase': 1,
    'ribosome': 1,
    'thyroglobulin': 1,
    'virus-like-particle': 1,
}