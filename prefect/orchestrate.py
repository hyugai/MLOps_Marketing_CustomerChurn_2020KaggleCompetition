# data structures
import numpy as np
import pandas as pd
# prefect
from prefect import flow, task
# others
from usr_modules.prefect_usr_defined import *

# load dataset
@task(name='Load Dataset')
@format_adjustment
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@task(name='Data Wrangling')
def data_wrangling(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return 

@flow
def main_flow() -> None:
    df_train = load_dataset(path='../dataset/raw/train.csv')

if __name__ == '__main__':
    main_flow()