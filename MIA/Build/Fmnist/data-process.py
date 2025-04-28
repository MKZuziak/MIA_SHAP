# preprocess_fashion_images.py
import os
import argparse
import pandas as pd
from PIL import Image
import boto3
from kaggle.api.kaggle_api_extended import KaggleApi

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--s3-bucket-name", type=str, required=True, help="S3 bucket name for storing processed images")
args = parser.parse_args()

# Constants
DATASET_NAME = "paramaggarwal/fashion-product-images-small"

SUBCATEGORIES = {"topwear", "bottomwear", "dress", "sandal", "shoes", "bags","innerwear","loungewear and nightwear","accessories","sports accessories"}

S3_BUCKET_NAME = args.s3_bucket_name
S3_PREFIX = "fashion-product-images-processed" # Replace with your desired S3 prefix

# Define SageMaker processing paths
LOCAL_DATA_DIR = "/opt/ml/processing/input/data"
LOCAL_PROCESSED_DIR = "/opt/ml/processing/output"
LOCAL_IMAGE_DIR = os.path.join(LOCAL_DATA_DIR, "images")

# Ensure directories exist
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
os.makedirs(LOCAL_IMAGE_DIR, exist_ok=True)
os.makedirs(LOCAL_PROCESSED_DIR, exist_ok=True)

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset
print("Downloading dataset from Kaggle...")
api.dataset_download_files(DATASET_NAME, path=LOCAL_DATA_DIR, unzip=True)

# Load metadata
metadata_path = os.path.join(LOCAL_DATA_DIR, "styles.csv")
df = pd.read_csv(metadata_path, error_bad_lines=False)
print(f"printing number of df before filtering  ......{df.shape}")
print("Column names:", df.columns)

# Filter for selected subcategories

# Convert subCategory to lowercase and remove extra spaces
df["subCategory"] = df["subCategory"].astype(str).str.strip().str.lower()

# Convert SUBCATEGORIES to lowercase
SUBCATEGORIES = {s.lower() for s in SUBCATEGORIES}

# Apply filtering
df_filtered = df[df["subCategory"].isin(SUBCATEGORIES)]
print("Unique subcategories in dataset:", df["subCategory"].unique())

print(f"Found {len(df_filtered)} images in the selected subcategories.")

# Initialize S3 client
s3_client = boto3.client("s3")

# Preprocess and upload images
print("Preprocessing and uploading images...")
missing_images = 0
print(f"printing number of df filtered ......{df_filtered.shape}")




LOCAL_DATA_DIR = "/opt/ml/processing/input/data"
print("Files in data directory:", os.listdir(LOCAL_DATA_DIR)[:10])  # Show first 10 files

for index, row in df_filtered.iterrows():
    print(f"step {index}")
    image_id = row["id"]
    image_path = os.path.join(LOCAL_IMAGE_DIR, f"{image_id}.jpg")
    
    # Check if the image exists
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_id}.jpg not found. Skipping.")
        missing_images += 1
        continue
    
    # Open and preprocess the image
    with Image.open(image_path) as img:
        img = img.convert("L")  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        
        # Create subcategory folder
        subcategory_folder = os.path.join(LOCAL_PROCESSED_DIR, row["subCategory"])
        os.makedirs(subcategory_folder, exist_ok=True)

        # Save the processed image inside the subcategory folder
        processed_image_path = os.path.join(subcategory_folder, f"{image_id}.jpg")
        img.save(processed_image_path)
                
        # Upload to S3
        s3_key = s3_key = f"{S3_PREFIX}/{row['subCategory']}/{image_id}.jpg"

        try:
            s3_client.upload_file(processed_image_path, S3_BUCKET_NAME, s3_key)
            print(f"✅ Successfully uploaded {s3_key} to S3.")
        except Exception as e:
            print(f"❌ Failed to upload {s3_key} due to: {e}")

        #s3_client.upload_file(processed_image_path, S3_BUCKET_NAME, s3_key)
        print(f"Uploaded {s3_key} to S3.")

print(f"Preprocessing complete! {missing_images} images were missing.")
