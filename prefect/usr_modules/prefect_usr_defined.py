# data structures
import numpy as np
import pandas as pd
# model selection
from sklearn.model_selection import train_test_split
# others
import functools

#
def adjust_format(func):
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
def handle_duplications(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> pd.DataFrame:
        df: pd.DataFrame = func(*args, **kargs)
        ## counts
        mask = df.duplicated()
        counts = mask.sum()
        print(f'Total duplications: {counts}')
        ##
        df.drop_duplicates(inplace=True)

        return df
    
    return wrapper

# 
def handle_single_value_columns(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> pd.DataFrame:
        df: pd.DataFrame = func(*args, **kargs)
        ## detection
        nuniques = df.nunique()
        single_value_names = nuniques[nuniques == 1].index.tolist()
        print(f'Single-value columns\'s name: {single_value_names}')
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

#
def split_dataset(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> tuple[pd.DataFrame, pd.DataFrame]:
        df: pd.DataFrame = func(*args, **kargs)
        ##
        column_names = df.columns
        X, y = df[column_names.drop('churn').tolist()].values, df['churn'].values
        ##
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.3, random_state=7, 
            stratify=y 
        )
        df_train = pd.DataFrame(
            data= np.hstack(tup=[X_train, y_train.reshape(-1, 1)]), 
            columns=column_names
        )
        df_test = pd.DataFrame(
            data=np.hstack(tup=[X_test, y_test.reshape(-1, 1)]), 
            columns=column_names
        )

        return df_train, df_test
    
    return wrapper

#
def validate_columns_compatible(func):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> None:
        df_train, df_test = func(*args, **kargs)
        ## detections
        column_names = dict(
            train=np.sort(df_train.columns.to_numpy()), 
            test=np.sort(df_test.columns.to_numpy())
        )
        try:
            mask = column_names['train'] != column_names['test']
            if np.sum(mask) != 0:
                for train_col_name, test_col_name in zip(column_names['train'][mask], column_names['test'][mask]):
                    print(f'{train_col_name} is not compatible with {test_col_name}')
            else:
                print('Columns compatile: Passed.')
        except:
            print(f'Please check the number of columns of each file again.')
        

        return df_train, df_test
    
    return wrapper