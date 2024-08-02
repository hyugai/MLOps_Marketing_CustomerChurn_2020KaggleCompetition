# data structures
import numpy as np
import pandas as pd
# prefect
from prefect import flow, task
# others
import os, sys
# decorators
cwd = os.getcwd()
os.chdir('../')
modules_path = os.getcwd()
if modules_path not in sys.path:
    sys.path.append(modules_path)
os.chdir(cwd)

from src.prefect.data_wrangling import *
from src.prefect.model_engineering import *

# load dataset
@task(name='Load Dataset', log_prints=False)
@adjust_format
def load_dataset(path: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    data_from_detections = dict()

    return df, data_from_detections

# subflow: data wrangling
@task(name='Detections', log_prints=False)
@detect_single_value_columns
@detect_duplications
@detect_missing_values
def detect(df: pd.DataFrame, data_from_detections: dict) -> tuple[pd.DataFrame, dict]:
    return df, data_from_detections

@task(name='Handling after detections')
@handle_single_value_columns
@handle_duplications
def handle(df: pd.DataFrame, data_from_detections: dict) -> tuple[pd.DataFrame, dict]:
    return df, data_from_detections

@flow(name='Subflow: Data Wrangling', log_prints=False)
def data_wrangling() -> tuple[pd.DataFrame, dict]:
    df, data_from_detections = load_dataset('../storage/data/raw/train.csv')
    df, data_from_detections = detect(df, data_from_detections)
    df, data_from_detections = handle(df, data_from_detections)
    
    return df, data_from_detections

# subflow: model engineering
@task(name='Prepare data to train', log_prints=False)
@get_selected_features
@split_dataset
def prepare_data_to_train(df: pd.DataFrame, artifacts_path: dict) -> tuple[pd.DataFrame, dict]:
    return df, artifacts_path

@task(name='Get optimized hyper-parameters', log_prints=False)
def get_optimized_hyp_params(train: np.ndarray) -> tuple[np.ndarray]:
    return train

@flow(name='Subflow: Model engineering', log_prints=False)
def model_engineering(df: pd.DataFrame) -> None:
    artifacts_path = dict(
        feature_selector='../storage/.notebook/ohe_quantiletransform.joblib', 
        model='../storage/temp/model.joblib'
    )
    train, test, artifacts_path = prepare_data_to_train(df, artifacts_path)

    return None

# main flow
@flow(name='Main flow', log_prints=False)
def main_flow() -> None:
    df, _ = data_wrangling()
    model_engineering(df=df)

    return None

if __name__ == '__main__':
    main_flow()