# data structures
import numpy as np
import pandas as pd
# others
import functools

#
def format_adjustment(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> pd.DataFrame:
        df: pd.DataFrame = func(*args, **kargs)
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

#
def detect_missing_values(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> pd.DataFrame:
        df: pd.DataFrame = func(*args, **kargs)
        ## counts
        mask = df.isnull()
        counts = mask.sum(axis=0)
        print(f'Total missing values per column: \n{counts}')
        
        return df

    return wrapper

#
def handling_duplications(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> pd.DataFrame:
        df: pd.DataFrame = func(*args, **kargs)
        ##counts
        mask = df.duplicated()
        counts = mask.sum()
        print(f'Total duplications: {counts}')
        ##
        df.drop_duplicates(inplace=True)

        return df
    
    return wrapper

# 
def handling_single_value_columns(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> pd.DataFrame:
        df: pd.DataFrame = func(*args, **kargs)
        ## detection
        nuniques = df.nunique()
        single_value_names = nuniques[nuniques == 1].index.tolist()
        print(f'Single value columns\'s name: {single_value_names}')
        ##
        try:
            df.drop(
                labels=single_value_names, axis=1, 
                inplace=True
            )
        except:
            print('Please check column\'s names again')

        return df
    
    return wrapper
    
