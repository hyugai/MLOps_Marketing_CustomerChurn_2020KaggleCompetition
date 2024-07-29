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
        df_format_adjusted.columns = [name.strip() for name in df.columns.tolist()]
        cat_cols = df.select_dtyeps('object')\
            .columns.tolist()
        ## cloning
        df_format_adjusted = df.copy()
        ## stripping 
        df_format_adjusted[cat_cols] = df_format_adjusted[cat_cols]\
            .map(lambda x: x.strip())
        ## null strings
        mask = (df_format_adjusted[cat_cols] == '')\
            .any(axis=0)
        null_names = mask[mask == True]\
            .index.tolist()
        df_format_adjusted[null_names] = df_format_adjusted[null_names]\
            .map(lambda x: np.nan if x == '' else x)

        return df_format_adjusted
    
    return wrapper