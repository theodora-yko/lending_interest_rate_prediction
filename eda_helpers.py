import os
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import openpyxl
import re as re
import seaborn as sns
import math

import numpy as np 
from sqlalchemy import create_engine
import matplotlib.cm as cm
import requests
import statsmodels.api as statsmdl
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings("ignore")


## general visualisation / tidying up 
def split_dtype_df(df_info, n_parts=3):
    total = len(df_info)
    chunk_size = (total + n_parts - 1) // n_parts  # ceil division
    return [df_info.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(n_parts)]

from IPython.display import display_html

def split_dtype_df_side_by_side(df_info, n_parts=3):
    # Display dataframe slices side by side in Jupyter
    total = len(df_info)
    chunk_size = (total + n_parts - 1) // n_parts
    dfs = [df_info.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(n_parts)]
    
    style = """
    <style>
        table {
            table-layout: fixed;
            width: 20%;
        }
        th, td {
            word-wrap: break-word;
            text-align: right;
        }
        .df-table {
            display: inline-block;
            width: 49%;
            vertical-align: top;
            padding-right: 1%;
        }
    </style>
    """

    html_tables = [
        df_chunk.to_html(index=False)
        for df_chunk in dfs
    ]
    
    styled_tables = [
        f'<div style="display: inline-block; vertical-align: top; padding: 0 15px;">{table}</div>'
        for table in html_tables
    ]
    
    display_html(''.join(styled_tables), raw=True)
    
def remove_space_from_columns(df): 
    df.columns = df.columns.str.replace(' ', '') 
    return df

def drop_empty_cols(df): 
    for col in df.columns: 
        if not len(df[col].dropna()): 
            print(f"No information in {col}, dropped")
            df.drop(columns=[col],inplace=True)
    return df

def remove_cols_wo_info(df1): 
    # remove all columns that only have a single value for all rows &. all unique rows 
    remove_cols = [] 
    for col in df1.select_dtypes(include=['object']).columns: 
        if df1[col].nunique() == 1 or df1[col].nunique() == df1.shape[0]:
            remove_cols.append(col)

    df1.drop(columns=remove_cols, inplace=True)
    return df1

## column specific processing 
# process all money-related columns 
def convertToFloat(val):
    try:
        clean_val = str(val).replace('$', '').replace(',', '')
        if '%' in clean_val:
            clean_val = str(clean_val).replace('%', '')
            if '.' in clean_val: 
                clean_val = clean_val/100        
        return round(float(clean_val), 0)
    except:
        return None

def process_numerical_cols(df1): 
    for col in df1.select_dtypes(include=['object']).columns: 
        if df1[col].dropna().empty or isinstance(df1[col].iloc[0], float):
            continue
        first_val = df1[col].dropna().iloc[0]
        if ('$' in first_val) or ('%' in first_val): 
            df1[col] = df1[col].apply(convertToFloat)
            print(f"Processed numerical column: {col}")
    return df1

def process_datecols(df1): 
    date_cols = df1.columns[df1.columns.str.lower().str.contains('date')]
    for col in date_cols: 
        try:
            df1[col] = pd.to_datetime(df1[col], errors='coerce')
            print(f"Processed date column: {col}")
        except:
            pass
    return df1

def initial_preprocessing(df):
    df = remove_space_from_columns(df)
    df = process_numerical_cols(df)
    df = remove_cols_wo_info(df)
    df = process_datecols(df)
    df = drop_empty_cols(df)
    return df

## exploration
def print_na_columns(df, nan_cutoff=90): 
    na_exists_cols = df.columns[df.isna().sum(axis=0) > 1]
    nan_percentage_df = df[na_exists_cols].isna().sum(axis=0).apply(lambda x: np.round(x/df.shape[0]*100,2)).sort_values(ascending=False)
    nan_percentage_df.columns = ['nan_percentage']
    nan_percentage_df = nan_percentage_df[nan_percentage_df > nan_cutoff]
    if nan_percentage_df.shape[0]: 
        print(f"------ columns that have nan values > {nan_cutoff}% ------")
        display(nan_percentage_df)
    majority_nan_cols = nan_percentage_df[nan_percentage_df>nan_cutoff].index
    return majority_nan_cols

def print_dtype_info(df, majority_nan_cols): 
    # print out all nonnull column types - categorical, cts, date, other (text for ex.)
    print(f"--------------- not null columns dtype -------------------")
    not_majority_nan_cols = df.columns[~df.columns.isin(majority_nan_cols)]
    data_type_df = pd.DataFrame(df[not_majority_nan_cols].dtypes, columns = ['Data Type']).reset_index()
    data_type_df.columns = ['Column Name', 'Data Type']
    split_dtype_df_side_by_side(data_type_df.sort_values(by=['Data Type','Column Name']), n_parts=4)
    
    print(f"--------------- not null continuous variables summary statistics -------------------")
    split_dtype_df_side_by_side(df.select_dtypes(include =['int64', 'float64', 'int32', 'float32']).describe().T.reset_index().sort_values(by='mean', ascending=False), n_parts=2)

    print(f"--------------- not null categorical variables summary statistics -------------------")
    split_dtype_df_side_by_side(df.select_dtypes(include=['object']).describe().T.reset_index().sort_values(by='unique', ascending=False), n_parts=2)
    
def initial_describe(df, nan_cutoff=90): 
    majority_nan_cols = print_na_columns(df, nan_cutoff)
    print_dtype_info(df, majority_nan_cols)
    

def select_categorical_columns(df, unique_cutoff = 50):
    cat_col_list = list(df.select_dtypes(include=['object']).columns)
    acceptable_cols = []
    for cat_col in cat_col_list: 
        nunique_col = df[cat_col].nunique()
        if nunique_col > unique_cutoff: 
            print(f'"{cat_col}" has {nunique_col} too many unique values, passed')
        else: 
            acceptable_cols.append(cat_col)
    return acceptable_cols


## data visualizations
def create_col_data_per_group(df, column, date_col = 'InjuryDate',target = 'HighAge1ToUltFactor'): 
    col_data = df[[column, target, date_col]]
    col_data[date_col] = col_data[date_col].dt.year
    return col_data

def process_data(col_data, column, target = 'HighAge1ToUltFactor', top_n = 10): 
    d1, d0 = col_data[col_data[target] == 1].dropna(), col_data[col_data[target] == 0].dropna()
    topN_d1 = d1.value_counts().index[:top_n]
    topN_d0 = d0.value_counts().index[:top_n]
    
    d1.loc[:, column] = d1[column].apply(lambda x: x if x in topN_d1 else 'Other')
    d0.loc[:, column] = d0[column].apply(lambda x: x if x in topN_d0 else 'Other')
    return d1, d0 

def draw_bar_plots_by_year(df, column, date_col = 'InjuryDate', target = 'HighAge1ToUltFactor'):
    fig, ax = plt.subplots(figsize=(10,5), nrows =1, ncols = 2)
    col_data = create_col_data_per_group(df, column)
    d1, d0 = process_data(col_data, column, target)
    create_barh_plot(ax[0], d0, column, date_col, which = 0)
    create_barh_plot(ax[1], d1, column, date_col, which = 1)
    fig.suptitle(column)
    fig.show()

def create_barh_plot(ax, subset_df, column, date_col, target, which):
    grouped = (
        subset_df.drop(columns=[target]).groupby([date_col, column])
        .size()
        .groupby(level=[0, 1])  # normalize per (year, target)
        .transform(lambda x: x / x.sum())
        .rename("Proportion")
        .reset_index()
    )
    df_pivot = grouped.pivot_table(index=[date_col], columns=column, values='Proportion', fill_value=0)
    df_pivot.sort_index(inplace=True)
    normalized = df_pivot.div(df_pivot.sum(axis=1), axis=0)
    normalized.plot(kind='barh', stacked=True, colormap='tab20', ax=ax)
    ax.set_title(f"{target} = {which}")

def create_heatmap(df, column_list, target='HighAge1ToUltFactor', date_col='InjuryDate'):
    ncols = 4
    nrows = math.ceil(len(column_list)/4)
    fig, ax = plt.subplots(figsize=(20, 4 * nrows), nrows=nrows, ncols=ncols)
    for column, ax_i in zip(column_list, ax.flatten()):
        col_data = create_col_data(df, column)
        pivot = col_data.pivot_table(index=column, columns=date_col, values=target)
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu', ax = ax_i)
    fig.show()


## feature generation 
def crtNumericCategories(df, sourceVar, targetVar, weight=None):
    temp_weight_col = '_temp_weight_'
    if weight is None:
        df[temp_weight_col] = 1
    else:
        df[temp_weight_col] = df[weight]

    # Step 1: Compute weighted mean of targetVar grouped by sourceVar
    weighted_means = (
        df.groupby(sourceVar).apply(
            lambda g: (g[targetVar] * g[temp_weight_col]).sum() / g[temp_weight_col].sum()
        ).reset_index(name='weighted_mean')
    )

    # Step 2: Rank the categories based on weighted mean (lowest = 1, highest = 4)
    weighted_means['rank'] = weighted_means['weighted_mean'].rank(method='dense', ascending=True).astype(int)

    # Step 3: Map back to the original dataframe
    mapping = dict(zip(weighted_means[sourceVar], weighted_means['rank']))
    
    # Print the mapping
    print(f"\nMapping of '{sourceVar}' to numeric categories ({sourceVar}N):")
    for k, v in mapping.items():
        print(f"  {k}: {v}")

    new_col_name = sourceVar.replace(' ', '') + 'N'
    df[new_col_name] = df[sourceVar].map(mapping)

    # Clean up temporary column
    if temp_weight_col in df.columns and weight is None:
        df.drop(columns=temp_weight_col, inplace=True)

    return df

