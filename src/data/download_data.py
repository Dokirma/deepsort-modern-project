import argparse
import os
import zipfile

import gdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--output", default="data/raw/mot_data.zip")
    parser.add_argument("--extract-to", default="data/mot")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.extract_to, exist_ok=True)

    url = f"https://drive.google.com/uc?id={args.file_id}"

    print("Downloading dataset...")
    gdown.download(url, args.output, quiet=False)

    print("Extracting dataset...")
    with zipfile.ZipFile(args.output, "r") as zip_ref:
        zip_ref.extractall(args.extract_to)

    print("Done.")


if __name__ == "__main__":
    main()
