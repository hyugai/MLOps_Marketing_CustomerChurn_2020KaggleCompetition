# data structures
import numpy as np
import pandas as pd
# model selection
from sklearn.model_selection import train_test_split
# mlflow
import mlflow
# others
import functools
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
