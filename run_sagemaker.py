import sagemaker
import argparse
from sagemaker.sklearn.estimator import SKLearn
import boto3
import os 
from dotenv import load_dotenv

from sagemaker.local import LocalSession

load_dotenv()
role = os.getenv("SAGEMAKER_ROLE_ARN")
bucket = os.getenv("SAGEMAKER_BUCKET")
region = boto3.Session().region_name

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True, help="Name of the data file in S3")
args = parser.parse_args()
input_path = f"s3://{bucket}/data/{args.filename}"

# Create the SKLearn estimator
sklearn_estimator = SKLearn(
    entry_point="train.py",
    source_dir=".",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    output_path=f"s3://{bucket}/output",
    base_job_name="interest-rate-model",
)

# Launch training job
sklearn_estimator.fit({"training": input_path},wait=True, logs=True)
