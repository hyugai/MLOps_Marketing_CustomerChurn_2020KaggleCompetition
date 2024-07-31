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

from usr_modules.notebook.features_engineering import SFS_OSP
# others
import functools, joblib
from collections.abc import Callable


# spllit dataset
def split(func: Callable[[pd.DataFrame], pd.DataFrame]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        df = func(*args, **kargs)
        columns_name = df.columns

        ## train-test split
        X, y = df[columns_name.drop('churn')].values, df['churn'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.3, random_state=7, 
            stratify=y
        )

        return X_train, X_test, y_train, y_test, columns_name.to_numpy()
    
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

# hyper-parameters optimization
def get_selected_features(func: Callable[[dict, np.ndarray, np.ndarray], tuple[dict, np.ndarray, np.ndarray]]):
    def wrapper(*args, **kagrs) -> tuple[dict, np.ndarray, np.ndarray]:
        artifacts_path, X_train, y_train = func(*args, **kagrs)
        feature_selector = joblib.load(artifacts_path['feature_selector'])
        selected_X_train = feature_selector.transform(X_train)

        return artifacts_path, selected_X_train, y_train
    
    return wrapper