# data structures
import numpy as np
import pandas as pd
# model selection
from sklearn.model_selection import train_test_split
# mlflow
import mlflow
# user-define modules
import os, sys
cwd = os.getcwd()
os.chdir('../../')
usr_modules_path = os.getcwd()
if usr_modules_path not in sys.path:
    sys.path.append(usr_modules_path)
os.chdir(cwd)

from usr_modules.notebook.features_engineering import *

# optuna
import optuna 
# others
import functools, joblib
from collections.abc import Callable


# spllit dataset
def split_dataset(func: Callable[[pd.DataFrame], pd.DataFrame]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> tuple[np.ndarray, np.ndarray, dict]:
        df, artifacts_path = func(*args, **kargs)
        columns_name = df.columns
        ## train-test split
        X, y = df[columns_name.drop('churn')].values, df['churn'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.3, random_state=7, 
            stratify=y
        )
        train = np.hstack([X_train, y_train.reshape(-1, 1)])
        test = np.hstack([X_test, y_test])
        ##
        artifacts_path['base_columns_name'] = columns_name.to_numpy()

        return train, test, artifacts_path
    
    return wrapper

def get_selected_features(func: Callable[[np.ndarray, np.ndarray, dict], tuple[np.ndarray, np.ndarray, dict]]):
    @functools.wraps(func)
    def wrapper(*args, **kagrs) -> tuple[np.ndarray, np.ndarray, dict]:
        train, test, artifacts_path = func(*args, **kagrs)
        X_train, y_train = train[:, :-1], train[:, -1]
        X_test, y_test = test[:, :-1], test[:, -1]
        ##
        feature_selector = joblib.load(artifacts_path['feature_selector'])
        selected_X_train = feature_selector.transform(X_train)
        selected_X_test = feature_selector.transform(X_test)
        ##
        train, test = np.hstack([selected_X_train, y_train.reshape(-1, 1)]), np.hstack([selected_X_test, y_test.reshape(-1, 1)])

        return train, test, artifacts_path
    
    return wrapper

# connect to local mlflow
def connect_local_mlflow(func: Callable[[str], str]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> None:
        experiment_name = func(*args, **kargs)

        ## set tracking uri
        mlflow.set_tracking_uri('http://127.0.0.1:5000')

        ## experiment
        try:
            mlflow.create_experiment(
                name=experiment_name, 
                artifact_location='.mlflow/.artifacts_store'
            )
            mlflow.set_experiment(
                experiment_name=experiment_name
            )
        except:
            mlflow.set_experiment(
                experiment_name=experiment_name
            )

        return None
    
    return wrapper


def objective_lgbm(trial: optuna.Trial):
    ##
    hyper_params = dict()
    hyper_params['max_depth'] = trial.suggest_int(
        name='max_depth', 
        low=0, high=10
    )
    hyper_params['num_leaves'] = trial.suggest_int(
        name='num_leaves', 
        low=31, high=127
    )
    hyper_params['min_data_in_leaf'] = trial.suggest_int(
        name='min_data_in_leaf', 
        low=10, high=200
    )

    lgbm = LGBMClassifier(verbose=-1, n_jobs=-1, **hyper_params)

    ##
    transformers = SFS_OSP(
        ohe=OneHotEncoder(drop='first', sparse_output=False), 
        scaling=[('scaling', QuantileTransformer(output_distribution='normal'))]
    )
    steps = [('transformers', transformers), 
             ('resampling', SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))), 
             ('LGBM', lgbm)]
    
    pipeline = Pipeline(steps)

def start_optimization(func: Callable[[dict, np.ndarray, np.ndarray], tuple[dict, np.ndarray, np.ndarray]]):
    @functools.wraps(func)
    def wrapper(*args, **kargs):
        artifacts_path, selected_X_train, y_train = func(*args, **kargs)
