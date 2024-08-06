from prefect import task, flow

from src.prefect.data_wrangling import *
from src.mlflow.mlflow_usr_defined import *
from src.notebook.features_engineering import *

import joblib
import pickle
import dill

class UnpicklerForSFS(pickle.Unpickler):
    def find_class(self, module_name: str, global_name: str):
        if module_name == '__main__':
            module_name = 'test'
        return super().find_class(module_name, global_name)

@task(log_prints=True)
def try_loadinng_joblib():
    #model = joblib.load('storage/temp/sfs.pkl')
    with open('storage/temp/sfs.pkl', 'rb') as input:
        model = dill.load(input)
    return model

@task(log_prints=True)
def try_customized_unpickler():
    with open('storage/temp/unpickler.pkl', 'rb') as f:
        unpickler = UnpicklerForSFS(f)
        model = unpickler.load()

    return model

@flow(log_prints=True)
def main_flow_v0():
    model = try_customized_unpickler()


@flow(log_prints=True)
def main_flow():
    model = try_loadinng_joblib()
    print(model)

if __name__ == '__main__':
    #main_flow.serve('joblib')
    main_flow_v0.serve('unpickler')