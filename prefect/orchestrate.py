# data structures
import numpy as np
import pandas as pd
# prefect
from prefect import flow, task
# others
from usr_modules.prefect_usr_defined import *

# load dataset
@task(name='Load Dataset', log_prints=False)
@adjust_format
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@task(name='Data Wrangling', log_prints=False)
@split_dataset
@handle_single_value_columns
@handle_duplications
@detect_missing_values
def data_wrangling(df: pd.DataFrame) -> pd.DataFrame:
    return df


@flow(log_prints=False)
def main_flow() -> None:
    df = load_dataset(path='../dataset/raw/train.csv')
    df_train, df_test = data_wrangling(df)

if __name__ == '__main__':
    main_flow()