from prefect import task, flow

from src.prefect.data_wrangling import *
from src.mlflow.mlflow_usr_defined import *
from src.notebook.features_engineering import *

import joblib
import dill

@task(log_prints=True)
def try_loadinng_joblib():
    #model = joblib.load('storage/temp/sfs.pkl')
    with open('storage/temp/sfs.pkl', 'rb') as input:
        model = dill.load(input)
    return model

@flow(log_prints=True)
def main_flow():
    model = try_loadinng_joblib()
    print(model)

if __name__ == '__main__':
    main_flow.serve('joblib')