# Deep-Learning-Project-CS

**Kaggle competition link:** [CryoET Object Identification](https://www.kaggle.com/competitions/czii-cryo-et-object-identification)

## Description

Protein complexes (such as oxygen-carrying hemoglobin, or keratin in hair, and thousands of others) are essential for cell function, and understanding their interactions is essential for our health and finding new disease treatments. Cryo-electron tomography (cryoET) creates 3D images—called tomograms—at near-atomic detail, showing proteins in their very complex and crowded natural environment. Therefore, cryoET has immense potential to unlock the mysteries of the cell.

There is a wealth of cryoET tomograms that is yet to be fully mined. A large and growing portion of this published corpus exists in a standardized format in the cryoET data portal (cryoetdataportal.czscience.com). Mining this data requires automatic identification of each protein molecule within these images. This problem has not been solved even for proteins that are identifiable by the human eye. A generalizable solution will reveal the “dark matter” of the cell, and will enable thousands of discoveries contributing to human health.

This competition challenges you to create ML algorithms that automatically annotate five classes of protein complexes within a curated real-world cryoET dataset.

## Data Download Instructions

To set up the data for this project, follow these steps:

### 1. Install the Kaggle CLI

Ensure you have the Kaggle CLI installed. If not, you can install it with the following command:

```
pip install kaggle
```

Refer to the official Kaggle API documentation for detailed installation instructions: [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api#installation).

### 2. Set Up Kaggle API Credentials

1. Log in to your Kaggle account and go to [Account Settings](https://www.kaggle.com/account).
2. Scroll down to the **API** section and click on **Create New API Token**.
3. This will download a `kaggle.json` file to your computer.
4. Move this file to the directory `~/.kaggle/` (create the directory if it doesn’t exist) and set the correct permissions:

```
mkdir -p ~/.kaggle
mv /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Run the Data Setup Script

Navigate to the root directory of the project and run the data setup script:

```
bash scripts/setup_data.sh
```

This script will:

- Download the competition data from Kaggle.
- Unzip and organize the data into the project’s folder structure:
  - `data/raw/`: Contains raw downloaded files.
  - `data/processed/train/`: Training data.
  - `data/processed/test/`: Test data.
  - `submissions/sample_submission.csv`: Sample submission file.
