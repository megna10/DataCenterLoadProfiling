import os
import gdown
from pathlib import Path


DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1Sk8i1we6PU4hCl4R1xI8K6xjtVHc4A8N"
)

PROJECT_ROOT = Path(__file__).resolve().parent

# A file that should exist after the download succeeds.
CHECK_FILE = (
    PROJECT_ROOT
    / "ai_training_files"
    / "metric_clean.csv"
)


def download_data():
    # Don't download everything again if the data is already present.
    if CHECK_FILE.exists():
        print("Data already exists. Skipping Google Drive download.")
        return

    print("Downloading data from Google Drive...")

    gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=str(PROJECT_ROOT),
        quiet=False,
        remaining_ok=True,
    )

    if not CHECK_FILE.exists():
        raise RuntimeError(
            "Google Drive download completed, but "
            "ai_training_files/metric_clean.csv was not found. "
            "Check the Google Drive folder structure."
        )

    print("Data download complete.")

def main():
    download_data()

if __name__ == "__main__":
    main()
