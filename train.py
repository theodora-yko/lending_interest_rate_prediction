import eda_helpers as eda
import argparse
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import subprocess
import sys
import os
import joblib

# req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='/opt/ml/input/data/training')
    parser.add_argument('--output-data-dir', type=str, default='/opt/ml/model')
    args = parser.parse_args()

    # List available files
    file_list = os.listdir(args.input_data)
    print("Available files:", file_list)

    # Assume one file
    file_path = os.path.join(args.input_data, file_list[0])
    df = pd.read_csv(file_path)
    print(df.shape)
    df = df.drop(columns = ['term',
        'installment', 'grade', 'sub_grade', 'issue_month',
        'initial_listing_status', 'disbursement_method',
        'balance', 'paid_total', 'paid_principal', 'paid_interest',
        'paid_late_fees', 'loan_amount'])
    df = eda.initial_preprocessing(df)
    eda.initial_describe(df)

    X = df.drop(columns=['loan_status', 'interest_rate']).fillna(0) # or whatever columns
    cat_list=  ['emp_title', 'state', 'loan_purpose', 'homeownership', 'verified_income', 'verification_income_joint', 'application_type']
    for col in cat_list: 
        X[col] = X[col].astype("category").cat.codes
    X = X.select_dtypes(exclude=['object'])
    y = np.log1p(df['interest_rate'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    baseline_model = LinearRegression()
    reg = baseline_model.fit(X_train, y_train)
    print(f"R^2: {reg.score(X_test, y_test)}")
    
    # Save the model
    joblib.dump(baseline_model, f'{args.output_data_dir}/model.joblib')

    print("Model is trained on the following features:")
    for col in X_train.columns:
        print("-", col)
