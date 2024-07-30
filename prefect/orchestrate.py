# data structures
import numpy as np
import pandas as pd
# prefect
from prefect import flow, task
# others
from usr_modules.data_wrangling import *

# load dataset
@task(name='Load Dataset', log_prints=False)
@adjust_format
def load_dataset(path: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    data_from_detections = dict()

    return df, data_from_detections

# data wrangling
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

@flow(name='Data Wrangling', log_prints=True)
def data_wrangling() -> tuple[pd.DataFrame, dict]:
    df, data_from_detections = load_dataset('../dataset/raw/train.csv')
    df, data_from_detections = detect(df, data_from_detections)
    df, data_from_detections = handle(df, data_from_detections)
    
    return df, data_from_detections

# model engineering
@task(name='Split dataset', log_prints=True)
@split_dataset
def split_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return df

if __name__ == '__main__':
    df, data_from_detections = data_wrangling()