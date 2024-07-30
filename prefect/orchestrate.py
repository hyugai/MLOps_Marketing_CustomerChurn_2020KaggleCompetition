# data structures
import numpy as np
import pandas as pd
# prefect
from prefect import flow, task
# others
from usr_modules.prefect_usr_defined import *

# load dataset
@task(name='Load Dataset', log_prints=False)
@format_adjustment
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@task(name='Data Wrangling', log_prints=False)
@handling_single_value_columns
@handling_duplications
@detect_missing_values
def data_wrangling(df: pd.DataFrame) -> pd.DataFrame:
    return df

@task(name='Compatible validation')
def compatible_validation(train: pd.DataFrame, test: pd.DataFrame):
    return train, test

@flow
def main_flow() -> None:
    df_train = load_dataset(path='../dataset/raw/train.csv')
    df_train = data_wrangling(df_train)

if __name__ == '__main__':
    main_flow()