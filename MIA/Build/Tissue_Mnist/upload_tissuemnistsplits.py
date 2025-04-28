import boto3
import os
import argparse

s3_client = boto3.client("s3")

def s3_file_exists(bucket, s3_path):
    """Check if a file exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=s3_path)
        return True  # File exists
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False  # File does not exist
        else:
            raise  # Some other error occurred

def upload_folder_to_s3(local_folder, s3_bucket, s3_folder):
    """Upload a local folder to S3, skipping existing files."""
    for root, _, files in os.walk(local_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder)
            s3_path = f"{s3_folder}/{relative_path}"

            if s3_file_exists(s3_bucket, s3_path):
                print(f"Skipping {file}, already exists in S3.")
                continue  # Skip if file exists

            s3_client.upload_file(local_path, s3_bucket, s3_path)
            print(f"Uploaded {file} to s3://{s3_bucket}/{s3_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-folder", type=str, required=True, help="Path to the local folder to upload")
    parser.add_argument("--s3-bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--s3-prefix", type=str, required=True, help="S3 prefix (destination folder)")

    args = parser.parse_args()

    upload_folder_to_s3(args.local_folder, args.s3_bucket, args.s3_prefix)
