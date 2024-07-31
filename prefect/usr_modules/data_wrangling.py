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

# data wrangling: adjust format
def adjust_format(func: Callable[[str], tuple[pd.DataFrame]]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> pd.DataFrame:
        df, data_from_detections = func(*args, **kargs)
        ##
        df.columns = [name.strip() for name in df.columns.tolist()]
        cat_cols = df.select_dtypes('object')\
            .columns.tolist()
        
        ## cloning
        df_cleaned = df.copy()

        ## stripping 
        df_cleaned[cat_cols] = df_cleaned[cat_cols]\
            .map(lambda x: x.strip())
        
        ## null strings
        mask = (df_cleaned[cat_cols] == '')\
            .any(axis=0)
        null_names = mask[mask == True]\
            .index.tolist()
        df_cleaned[null_names] = df_cleaned[null_names]\
            .map(lambda x: np.nan if x == '' else x)
        
        ## meta data
        data_from_detections['adjust_format'] = null_names

        return df_cleaned, data_from_detections
    
    return wrapper

# data wrangling: missing values
def detect_missing_values(func: Callable[[pd.DataFrame, dict], tuple[pd.DataFrame, dict]]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> tuple[pd.DataFrame, dict]:
        df, data_from_detections = func(*args, **kargs)

        ## counts
        mask = df.isnull()
        counts = mask.sum(axis=0)
        print(f'Total missing values per column: \n{counts}')

        ## meta data
        data_from_detections['missing_values'] = counts
        
        return df, data_from_detections

    return wrapper

def handle_missing_values() -> None:
    return None

# data wrangling: duplications
def detect_duplications(func: Callable[[pd.DataFrame, dict], tuple[pd.DataFrame, dict]]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> tuple[pd.DataFrame, dict]:
        df, data_from_detections = func(*args, **kargs)

        ## counts
        mask = df.duplicated()
        counts = mask.sum()
        print(f'Total duplications: {counts}')

        ## meta data
        data_from_detections['duplications'] = counts

        return df, data_from_detections
    
    return wrapper

def handle_duplications(func: Callable[[pd.DataFrame, dict], tuple[pd.DataFrame, dict]]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> tuple[pd.DataFrame, dict]:
        df, data_from_detections = func(*args, **kargs)
        
        ## drop duplications
        df.drop_duplicates(inplace=True)
        print('Duplcations: Passed.')

        return df, data_from_detections
    
    return wrapper

# data wrangling: single-value columns
def detect_single_value_columns(func: Callable[[pd.DataFrame, dict], tuple[pd.DataFrame, dict]]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> tuple[pd.DataFrame, dict]:
        df, data_from_detections = func(*args, **kargs)
        ## detection
        nuniques = df.nunique()
        single_value_names = nuniques[nuniques == 1].index.tolist()
        print(f'Single-value columns\'s name: {single_value_names}')

        ## meta data
        data_from_detections['single_value_columns'] = single_value_names

        return df, data_from_detections
    
    return wrapper

def handle_single_value_columns(func: Callable[[pd.DataFrame, dict], tuple[pd.DataFrame, dict]]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> tuple[pd.DataFrame, dict]:
        df, data_from_detections = func(*args, **kargs)

        ## drop single-value columns
        try:
            df.drop(
                labels=data_from_detections['single_value_columns'], axis=1, 
                inplace=True
            )
            print('Single-value columns: Passed.')
        except:
            print(f"Please check the name of columns again: \n{data_from_detections['single_value_columns']}")

        return df, data_from_detections
    
    return wrapper

# model engineering: spllit dataset
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

# model engineering: connect to local mlflow
def connect_local_mlflow(func: Callable[[str], str]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> None:
        experiment_name = func(*args, **kargs)

        ## 
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
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
