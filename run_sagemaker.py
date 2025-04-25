import os 
from dotenv import load_dotenv
import argparse
from sagemaker.sklearn.estimator import SKLearn
import boto3
from sagemaker.sklearn.model import SKLearnModel
import numpy as np

load_dotenv()
role = os.getenv("SAGEMAKER_ROLE_ARN")
bucket = os.getenv("SAGEMAKER_BUCKET")
region = boto3.Session().region_name
model_name = "interest-rate-model"
endpoint_name = "interest-rate-predictor"

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, required=True, help="Name of the data file in S3")
parser.add_argument("--test_data", type=str, help="Path to data input *CSV file* for the model to run predictions on")
args = parser.parse_args()
input_path = f"s3://{bucket}/data/{args.filename}"
# test_data = args.test_data

# Create the SKLearn estimator
print("Start training")
sklearn_estimator = SKLearn(
    entry_point="train.py",
    source_dir=".",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    output_path=f"s3://{bucket}/output",
    base_job_name=model_name,
)

# Launch training job
sklearn_estimator.fit(
    {"training": input_path},
    job_name=model_name,
    wait=True,
    logs=True
)

print("Training completed")
print("Training job name:", sklearn_estimator.latest_training_job.name)
print("Model artifact S3 path:", sklearn_estimator.model_data)

# Deploy to an endpoint
jobname = sklearn_estimator.latest_training_job.name
model_path = f's3://{bucket}/output/{jobname}/output/model.tar.gz'

model = SKLearnModel(
    job_name="interest-rate-model",
    model_data=model_path,
    role=role,
    entry_point="inference.py",  
    framework_version="1.2-1",  
    py_version="py3",
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",  
    endpoint_name=endpoint_name  
)

# # === Predict ===
# user_input_list = list(map(float, args.sample_input.split(",")))
# sample_input = np.array([user_input_list])  # shape must be (1, num_features)

# prediction = predictor.predict(test_data)
# print("Predicted interest rate:", prediction)
predictor.delete_endpoint()