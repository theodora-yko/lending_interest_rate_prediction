import eda_helpers as eda
import argparse
import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor, XGBClassifier
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
    
    # pre-processing the data 
    print("Preprocessing the data")
    df = df.drop(columns = ['term',
       'installment', 'grade', 'sub_grade', 'issue_month',
       'initial_listing_status', 'disbursement_method',
       'balance', 'paid_total', 'paid_principal', 'paid_interest',
       'paid_late_fees', 'tax_liens'])    
    
    df['interest_rate'] = np.log1p(df['interest_rate'])
    risky_statuses = [
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Charged Off",
    'In Grace Period'
    ]

    df["is_risky"] = df["loan_status"].isin(risky_statuses)
    df.drop(columns = ['loan_status'], inplace = True)
    df = eda.remove_cols_wo_info(df)
    
    df['annual_income'] = df.apply(
    lambda row: max(row['annual_income'], row['annual_income_joint']) 
    if row['verification_income_joint'] in ['Verified', 'Source Verified']
    else row['annual_income'],
    axis=1
    )
    df['debt_to_income'] = df.apply(
        lambda row: max(row['debt_to_income'], row['debt_to_income_joint']) 
        if row['verification_income_joint'] in ['Verified', 'Source Verified']
        else row['annual_income'],
        axis=1
    )

    df.drop(columns=['annual_income_joint', 'debt_to_income_joint', 'verification_income_joint'], inplace=True)
    df = eda.initial_preprocessing(df)
    
    # imputing missing values for numerical columns
    imputer = SimpleImputer(strategy='median')
    cts_df = df.select_dtypes(exclude=['object'])
    null_cts_columns = cts_df.columns[cts_df.isnull().sum() > 0]
    null_cts_columns = list(set(null_cts_columns) - set(['is_risky', 'interest_rate', 'loan_amount']))
    df[null_cts_columns] = imputer.fit_transform(df[null_cts_columns])

    # Convert categorical columns to category type
    encoder = OneHotEncoder(drop='if_binary', sparse_output=False,handle_unknown='ignore')
    cat_df = df.select_dtypes(include=['object'])
    categorical_cols = cat_df.columns
    encoder.fit(df[categorical_cols])
    df_encoded = encoder.transform(df[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    df_encoded_df = pd.DataFrame(df_encoded, columns=encoded_cols, index=df.index)
    encoded_X_train = pd.concat([df.drop(columns=categorical_cols), df_encoded_df], axis=1)
    
    # XGB classification model 
    print("Training the classification model")
    classification_cols = ['loan_purpose_house', 'months_since_last_credit_inquiry', 'income_per_account']
    df_class = df[classification_cols]
    print("Classification Model is trained on the following features:")
    for col in classification_cols:
        print("-", col)
    y_class = df['is_risky']
    
    neg = (y_class == 0).sum()['is_risky']
    pos = (y_class == 1).sum()['is_risky']
    scale = neg / pos

    xgb_model = XGBClassifier(
    n_estimators=1000,            # start high, use early stopping
    learning_rate=0.05,           # slower learning
    max_depth=6,                  # tree depth
    scale_pos_weight=scale,       # important for imbalance
    subsample=0.8,                # row sampling
    colsample_bytree=0.8,         # feature sampling
    random_state=42,
    n_jobs=-1,
    verbosity=0
    )
    
    xgb_model.fit(
    df_class, y_class,
    eval_metric='auc',
    early_stopping_rounds=50,     # stop if no improvement
    verbose=True
    )
   
    # XGB regression model - hyper parameter tuned 
    print("Training the regression model")
    regression_cols = ['total_debit_limit', 'credit_util_ratio', 'debt_per_limit',
       'time_since_first_credit', 'total_credit_limit',
       'num_total_cc_accounts', 'num_mort_accounts', 'high_inquiry_flag',
       'avg_credit_limit_per_account', 'open_credit_ratio',
       'earliest_credit_line', 'inquiries_last_12m',
       'account_never_delinq_percent', 'no_deliquency_flag',
       'verified_income_Verified', 'homeownership_RENT',
       'application_type_joint', 'homeownership_MORTGAGE',
       'verified_income_Not Verified', 'accounts_opened_24m',
       'months_since_90d_late', 'months_since_90d_late_exists',
       'months_since_last_credit_inquiry', 'loan_purpose_vacation',
       'loan_purpose_credit_card', 'risky_debt_combo',
       'num_historical_failed_to_pay', 'num_satisfactory_accounts',
       'public_record_bankrupt', 'emp_title_owner',
       'months_since_last_delinq_exists', 'risk_score']
    df_reg = df[regression_cols]
    df_reg['risk_score'] = xgb_model.predict_proba(df_class)[:, 1]
    y_reg = df['interest_rate']
    
    print("Regression Model is trained on the following features:")
    for col in regression_cols:
        print("-", col)

    xgb_reg = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        min_child_weight = 7,
        max_depth=5,
        subsample=0.6,
        colsample_bytree=0.6,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
 
    xgb_reg.fit(
        df_reg, y_reg,
        eval_metric='rmse',
        early_stopping_rounds=50,
        verbose=True
    )
    
    print("Training completed, saving the model...")
    joblib.dump(xgb_reg, f'{args.output_data_dir}/model.joblib')
    
    print("Model saved successfully.")

