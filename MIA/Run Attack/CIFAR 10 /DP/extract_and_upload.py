import argparse
import boto3
import tarfile
import io
import os

# Initialize S3 client
s3_client = boto3.client("s3")

def list_s3_files(bucket, prefix):
    """List all files under an S3 prefix."""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]

def multipart_upload(s3_bucket, s3_key, file_obj, part_size=8 * 1024 * 1024):
    """Uploads a file to S3 using multipart upload."""
    try:
        # Step 1: Initiate multipart upload
        response = s3_client.create_multipart_upload(Bucket=s3_bucket, Key=s3_key)
        upload_id = response["UploadId"]
        parts = []
        part_number = 1

        while True:
            chunk = file_obj.read(part_size)
            if not chunk:
                break  # No more data to read
            
            part = io.BytesIO(chunk)

            # Step 2: Upload each part
            upload_response = s3_client.upload_part(
                Bucket=s3_bucket,
                Key=s3_key,
                PartNumber=part_number,
                UploadId=upload_id,
                Body=part
            )
            
            parts.append({"PartNumber": part_number, "ETag": upload_response["ETag"]})
            part_number += 1

        # Step 3: Complete multipart upload
        s3_client.complete_multipart_upload(
            Bucket=s3_bucket,
            Key=s3_key,
            UploadId=upload_id,
            MultipartUpload={"Parts": parts}
        )
        print(f"Successfully uploaded {s3_key} using multipart upload.")

    except Exception as e:
        print(f"Multipart upload failed for {s3_key}: {str(e)}")
        s3_client.abort_multipart_upload(Bucket=s3_bucket, Key=s3_key, UploadId=upload_id)
        raise

def stream_extract_and_upload(s3_bucket, tar_key, extracted_prefix):
    """Extract a .tar.gz file from S3 and upload extracted contents using multipart upload."""
    print(f"Processing {tar_key}...")

    # Download tar.gz file into memory
    tar_stream = io.BytesIO()
    s3_client.download_fileobj(s3_bucket, tar_key, tar_stream)
    tar_stream.seek(0)

    # Open tar archive
    with tarfile.open(fileobj=tar_stream, mode="r:gz") as tar:
        for member in tar:
            if member.isfile():  # Extract only files
                extracted_key = f"{extracted_prefix}/{member.name}"
                
                # Check if the file already exists in S3
                existing_files = list_s3_files(s3_bucket, extracted_key)
                if extracted_key in existing_files:
                    print(f"Skipping {extracted_key}, already exists.")
                    continue  # Skip existing files

                # Extract file
                file_obj = tar.extractfile(member)
                if file_obj:
                    print(f"Uploading {extracted_key}...")
                    multipart_upload(s3_bucket, extracted_key, file_obj)

    print(f"Extraction of {tar_key} completed.")

def process_all_tar_files(bucket, s3_prefix, extracted_prefix):
    """Process all .tar.gz files in the given S3 bucket prefix."""
    tar_files = list_s3_files(bucket, s3_prefix)
    tar_files = [f for f in tar_files if f.endswith(".tar.gz")]

    print(f"Found {len(tar_files)} .tar.gz files to process.")
    print(tar_files)
    
    #for tar_file in tar_files:
    tar_file = "datasets/datasets_tissuemnist_5.tar.gz"
    
    stream_extract_and_upload(bucket, tar_file, extracted_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and upload .tar.gz files from S3.")
    parser.add_argument("--input_s3", type=str, required=True, help="S3 prefix where .tar.gz files are stored")
    parser.add_argument("--output_s3", type=str, required=True, help="S3 prefix where extracted files should be uploaded")
    parser.add_argument("--bucket", type=str, required=True, help="S3 bucket name")
    args = parser.parse_args()

    process_all_tar_files(args.bucket, args.input_s3, args.output_s3)
