# Chemin de base pour ExperimentRuns
DATA_DIR = "data/raw/train/static/ExperimentRuns"
TARGET_DIR = "data/raw/train/overlay/ExperimentRuns"
SRC_DIR = "src"

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