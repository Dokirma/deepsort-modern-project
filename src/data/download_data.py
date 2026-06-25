import argparse
import os
import zipfile
import tarfile

import gdown


def extract_archive(archive_path, extract_to):
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_to)

    elif archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(extract_to)

    else:
        raise ValueError(f"Unknown archive format: {archive_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--output", default="data/raw/dlcv-final-project-videos.tar.gz")
    parser.add_argument("--extract-to", default="data/mot")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.extract_to, exist_ok=True)

    url = f"https://drive.google.com/uc?id={args.file_id}"

    if not os.path.exists(args.output):
        print("Downloading dataset...")
        gdown.download(url, args.output, quiet=False)
    else:
        print(f"Archive already exists: {args.output}")

    print("Extracting dataset...")
    extract_archive(args.output, args.extract_to)

    print("Done.")


if __name__ == "__main__":
    main()
