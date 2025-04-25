import joblib
import os
import pandas as pd
from io import StringIO

def model_fn(model_dir):
    """Load the trained model."""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

def input_fn(request_body, request_content_type):
    """Deserialize the request into a Pandas DataFrame."""
    if request_content_type == 'text/csv':
        return pd.read_csv(StringIO(request_body))
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Apply model to the input_data."""
    return model.predict(input_data)

def output_fn(prediction, content_type):
    """Serialize prediction into the response format."""
    if content_type == 'text/csv':
        return ','.join(map(str, prediction))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
