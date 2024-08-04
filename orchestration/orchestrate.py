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

"""
The "materials" variable include: 
    df: pd.DataFrame,
    info: dict,
    X_train && X_test && y_train && y_test: np.ndarray,
    artifacts_path: dict
    pipeline: Pipeline, 
    avg_fbeta: float,
    params: dict, 
    val_fbeta: float
"""

# subflow: data wrangling
@task(name='Detections', log_prints=False)
@detect_single_value_columns
@detect_duplications
@detect_missing_values
@adjust_format
def detect(path: str) -> dict:
    df = pd.read_csv(path)
    materials = {
        'df': df, 
        'info': dict()
    }

    return materials

@task(name='Handling after detections')
@handle_single_value_columns
@handle_duplications
def handle(materials: dict) -> dict:

    return materials

@flow(name='Subflow: Data Wrangling', log_prints=False)
def data_wrangling() -> tuple[pd.DataFrame, dict]:
    materials = detect('../storage/data/raw/train.csv')
    materials = handle(materials)

    return materials

# subflow: model engineering
@task(name='Prepare data to train', log_prints=False)
@get_selected_features
@split_dataset
def prepare_data_to_train(materials: dict) -> dict:
    
    return materials

@task(name='Get optimized hyper-parameters', log_prints=False)
@log_model
@tune_hyp_params
def optimize_model(materials: dict) -> dict:
    
    return materials

@flow(name='Subflow: Model engineering', log_prints=False)
def model_engineering(materials: dict) -> None:
    materials = prepare_data_to_train(materials)
    materials = optimize_model(materials)

    return None

# main flow
@flow(name='Main flow', log_prints=False)
def main_flow() -> None:
    materials = data_wrangling()
    materials['artifacts_path'] = dict(
        feature_selector='../storage/.notebook/ohe_quantiletransform.joblib',
        model='../storage/temp/model.joblib'
    )
    model_engineering(materials)

    return None

if __name__ == '__main__':
    main_flow()