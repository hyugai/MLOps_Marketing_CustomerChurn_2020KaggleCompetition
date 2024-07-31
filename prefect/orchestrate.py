# data structures
import numpy as np
import pandas as pd
# prefect
from prefect import flow, task
# decorators
from prefect_usr_modules.data_wrangling import *
from prefect_usr_modules.model_engineering import *

# load dataset
@task(name='Load Dataset', log_prints=True)
@adjust_format
def load_dataset(path: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    data_from_detections = dict()

    return df, data_from_detections

# subflow: data wrangling
@task(name='Detections', log_prints=True)
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

@flow(name='Subflow: Data Wrangling', log_prints=True)
def data_wrangling() -> tuple[pd.DataFrame, dict]:
    df, data_from_detections = load_dataset('../dataset/raw/train.csv')
    df, data_from_detections = detect(df, data_from_detections)
    df, data_from_detections = handle(df, data_from_detections)
    
    return df, data_from_detections

# subflow: model engineering
@task(name='Train-Test split', log_prints=True)
@split
def prepare_TrainTest_data(df: pd.DataFrame) -> pd.DataFrame:
    return df

@task(name='Connect to MLflow', log_prints=True)
@connect_local_mlflow
def set_experiment(experiment_name: str) -> str:
    return experiment_name

@task(name='Hyper-parameters opmization')
def optimize_hyper_params(artifacts_path: dict, X_train: np.ndarray, y_train: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    return artifacts_path

@flow(name='Subflow: Model engineering', log_prints=True)
def model_engineering(df: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test, columns_name = prepare_TrainTest_data(df)
    set_experiment('Model engineering')

# main flow
@flow(name='Main flow', log_prints=True)
def main_flow() -> None:
    df, data_from_detections = data_wrangling()
    model_engineering(df=df)

    return None

if __name__ == '__main__':
    main_flow()