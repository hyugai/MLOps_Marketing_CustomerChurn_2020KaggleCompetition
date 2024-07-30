# data structures
import numpy as np
import pandas as pd
# others
import functools
from collections.abc import Callable

# adjust format
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

# missing values
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

# duplications
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

def handle_duplications(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs):
        df: pd.DataFrame = func(*args, **kargs)

        return df.drop_duplicates(inplace=True)
    
    return wrapper

# single-value columns
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

def handle_single_value_columns(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> pd.DataFrame:
        df: pd.DataFrame = func(*args, **kargs)

