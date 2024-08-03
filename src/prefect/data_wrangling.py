# data structures
import numpy as np
import pandas as pd
# others
import functools
from collections.abc import Callable

# data wrangling: adjust format
def adjust_format(func: Callable[[str], dict]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> dict:
        materials = func(*args, **kargs)
        ##
        materials['df'].columns = [name.strip() for name in materials['df'].columns.tolist()]
        cat_cols = materials['df'].select_dtypes('object')\
            .columns.tolist()
        ## cloning
        df_cleaned = materials['df'].copy()
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
        ##
        materials['df'] = df_cleaned

        return materials
    
    return wrapper

# data wrangling: missing values
def detect_missing_values(func: Callable[[str], dict]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> dict:
        materials = func(*args, **kargs)
        ## counts
        mask = materials['df'].isnull()
        counts = mask.sum(axis=0)
        print(f'Total missing values per column: \n{counts}')
        ## meta data
        materials['info']['missing_values'] = counts
        
        return materials

    return wrapper

def handle_missing_values() -> None:
    return None

# data wrangling: duplications
def detect_duplications(func: Callable[[str], dict]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> dict:
        materials = func(*args, **kargs)
        ## counts
        mask = materials['df'].duplicated()
        counts = mask.sum()
        print(f'Total duplications: {counts}')
        ## meta data
        materials['info']['duplications'] = counts

        return materials
    
    return wrapper

def handle_duplications(func: Callable[[str], dict]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> dict:
        materials = func(*args, **kargs)
        ## drop duplications
        materials['df'].drop_duplicates(inplace=True)
        print('Duplcations: Passed.')

        return materials
    
    return wrapper

# data wrangling: single-value columns
def detect_single_value_columns(func: Callable[[str], dict]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> dict:
        materials = func(*args, **kargs)
        ## detection
        nuniques = materials['df'].nunique()
        single_value_names = nuniques[nuniques == 1].index.tolist()
        print(f'Single-value columns\'s name: {single_value_names}')
        ## meta data
        materials['info']['single_value_columns'] = single_value_names

        return materials
    
    return wrapper

def handle_single_value_columns(func: Callable[[str], dict]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> dict:
        materials = func(*args, **kargs)
        ## drop single-value columns
        try:
            materials['df'].drop(
                labels=materials['info']['single_value_columns_name'], axis=1, 
                inplace=True
            )
            print('Single-value columns: Passed.')
        except:
            print(f"Please check the name of columns again: \n{materials['info']['single_value_columns']}")

        return materials
    
    return wrapper

