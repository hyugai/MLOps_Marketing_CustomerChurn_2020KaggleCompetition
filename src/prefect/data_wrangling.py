# data structures
import numpy as np
import pandas as pd
# others
import functools
from collections.abc import Callable

# data wrangling: adjust format
def adjust_format(func: Callable[[str], pd.DataFrame]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> pd.DataFrame:
        df = func(*args, **kargs)
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

        return df_cleaned
    
    return wrapper

# data wrangling: missing values
def detect_missing_values(func: Callable[[pd.DataFrame], tuple[dict]]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> dict:
        materials = func(*args, **kargs)
        ## counts
        mask = materials['df_base'].isnull()
        counts = mask.sum(axis=0)
        print(f'Total missing values per column: \n{counts}')
        ## meta data
        materials['info']['total_missing_values'] = counts
        
        return materials

    return wrapper

def handle_missing_values() -> None:
    return None

# data wrangling: duplications
def detect_duplications(func: Callable[[pd.DataFrame], dict]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> dict:
        materials = func(*args, **kargs)
        ## counts
        mask = materials['df_base'].duplicated()
        counts = mask.sum()
        print(f'Total duplications: {counts}')
        ## meta data
        materials['info']['total_duplications'] = counts

        return materials
    
    return wrapper

def handle_duplications(func: Callable[[pd.DataFrame], dict]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> tuple[pd.DataFrame, dict]:
        materials = func(*args, **kargs)
        ## drop duplications
        materials['df_base'].drop_duplicates(inplace=True)
        print('Duplcations: Passed.')

        return materials
    
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



