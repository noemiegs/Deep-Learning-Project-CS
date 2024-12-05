#!/bin/bash

# Stop the script if any command fails
set -e

# Get the root directory of the Git project
PROJECT_ROOT=$(git rev-parse --show-toplevel)

# Kaggle competition name
COMPETITION_NAME="czii-cryo-et-object-identification"

# Define paths within the project structure
RAW_DATA_DIR="$PROJECT_ROOT/data/raw"
PROCESSED_DATA_DIR="$PROJECT_ROOT/data/processed"
SUBMISSIONS_DIR="$PROJECT_ROOT/submissions"

# Check if Kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle CLI is not installed. Please install it before running this script."
    echo "See: https://github.com/Kaggle/kaggle-api#installation"
    exit 1
fi

# Check if kaggle.json authentication file exists
KAGGLE_AUTH_FILE="$HOME/.kaggle/kaggle.json"
if [[ ! -f "$KAGGLE_AUTH_FILE" ]]; then
    echo "Error: Kaggle authentication file ($KAGGLE_AUTH_FILE) is missing."
    echo "Download kaggle.json from your Kaggle account and place it in ~/.kaggle/ with the correct permissions."
    echo "See: https://github.com/Kaggle/kaggle-api#api-credentials"
    exit 1
fi

# Ensure the kaggle.json file has secure permissions
if [[ $(stat -c "%a" "$KAGGLE_AUTH_FILE") != "600" ]]; then
    echo "Adjusting permissions for $KAGGLE_AUTH_FILE for security."
    chmod 600 "$KAGGLE_AUTH_FILE"
fi

# Create the project directories if they don't exist
echo "Creating project directories under $PROJECT_ROOT..."
mkdir -p "$RAW_DATA_DIR" "$PROCESSED_DATA_DIR" "$SUBMISSIONS_DIR"

# Download data from Kaggle
ZIP_FILE="$RAW_DATA_DIR/$COMPETITION_NAME.zip"
echo "Downloading data from Kaggle..."
kaggle competitions download -c "$COMPETITION_NAME" -p "$RAW_DATA_DIR"

# Unzip the data
echo "Unzipping data..."
unzip -q "$ZIP_FILE" -d "$RAW_DATA_DIR"

# Organize files into their respective directories
echo "Organizing files..."
if [[ -d "$RAW_DATA_DIR/train" && -d "$RAW_DATA_DIR/test" && -f "$RAW_DATA_DIR/sample_submission.csv" ]]; then
    echo "Moving sample submission to its directory..."
    mv "$RAW_DATA_DIR/sample_submission.csv" "$SUBMISSIONS_DIR/sample_submission.csv"
else
    echo "Error: Unexpected structure in the zip file. Please check its contents."
    exit 1
fi

# Clean up temporary files
echo "Cleaning up temporary files..."
rm "$ZIP_FILE"

# Confirmation message
echo "Data successfully downloaded, unzipped, and organized in $PROJECT_ROOT!"
