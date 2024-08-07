# data structures
import numpy as np
import pandas as pd
# model selection
from sklearn.model_selection import train_test_split
# mlflow
import mlflow
from mlflow.models import infer_signature
#user-define modules
import os, sys
cwd = os.getcwd()
os.chdir('../..')
usr_modules_path = os.getcwd()
if usr_modules_path not in sys.path:
    sys.path.append(usr_modules_path)
os.chdir(cwd)

from src.mlflow.mlflow_usr_defined import *
from src.notebook.features_engineering import *

# optuna
import optuna 
# others
import functools, joblib, pickle
from collections.abc import Callable

# customized Unpicler
class UnpicklerSFS(pickle.Unpickler):
    def find_class(self, module_name: str, global_name: str):
        if module_name == '__main__':
            module_name == 'orchestrate'

        return super().find_class(module_name, global_name)

# spllit dataset
def split_dataset(func: Callable[[dict], dict]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> dict:
        materials = func(*args, **kargs)
        columns_name = materials['df'].columns
        ## train-test split
        X, y = materials['df'][columns_name.drop('churn')].values, materials['df']['churn'].values
        materials['X_train'], materials['X_test'], materials['y_train'], materials['y_test'] = train_test_split(
            X, y, 
            test_size=0.3, random_state=7, 
            stratify=y
        )

        return materials
    
    return wrapper

# get selected features
def get_selected_features(func: Callable[[dict], dict]):
    @functools.wraps(func)
    def wrapper(*args, **kagrs) -> dict:
        materials = func(*args, **kagrs)
        ##
        with open(materials['artifacts_path']['feature_selector'], 'rb') as input:
            unpickler = UnpicklerSFS(input)
            feature_selector = unpickler.load()
        #feature_selector: SequentialFeatureSelector = joblib.load(materials['artifacts_path']['feature_selector'])
        materials['X_train'] = feature_selector.transform(materials['X_train'])
        materials['X_test'] = feature_selector.transform(materials['X_test'])

        return materials
    
    return wrapper

# objective function for LGBM 
def objecttive_lgbm(trial: optuna.Trial, materials: dict):
    params = dict()
    params['num_leaves'] = trial.suggest_int(
        name='LGBM__num_leaves', 
        low=31, high=127
    )
    params['max_depth'] = trial.suggest_int(
        name='LGBM__max_depth', 
        low=1, high=10
    )
    params['min_data_in_leaf'] = trial.suggest_int(
        name='LGBM__min_data_in_leaf', 
        low=20, high=100
    )
    lgbm = LGBMClassifier(verbose=-1, n_jobs=-1, **params)
    ##
    transformers = SFS_OSP(
        ohe=OneHotEncoder(drop='first', sparse_output=False), 
        scaling=[('scaling', QuantileTransformer(output_distribution='normal'))]
    )
    pipeline = Pipeline(steps=[('transformers', transformers), 
                               ('resampling', SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))), 
                               ('LGBM', lgbm)])
    materials['pipeline'] = pipeline
    ## 
    kfold_result = cross_val_score(
        estimator=pipeline, 
        X=materials['X_train'], y=materials['y_train'], 
        cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3), 
        scoring=make_scorer(fbeta_score, beta=2)
    )

    return kfold_result.mean()

# tuning hyper-parameters
def tune_hyp_params(func: Callable[[dict], dict]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> dict:
        materials = func(*args, **kargs)
        le = LabelEncoder()
        materials['y_train'] = le.fit_transform(materials['y_train'])
        materials['y_test'] = le.transform(materials['y_test'])
        ##
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objecttive_lgbm(trial, materials), n_trials=5)
        ##
        materials['avg_fbeta'] = study.best_trial.value
        materials['params'] = study.best_params  
        ##
        materials['pipeline'].set_params(**materials['params'])
        materials['pipeline'].fit(materials['X_train'], materials['y_train'])
        dump_model(
            model=materials['pipeline'], 
            path=materials['artifacts_path']['model']
        )

        return materials
    
    return wrapper

# connect to local mlflow
def connect_local_mlflow(func: Callable[[dict], dict]):
    @functools.wraps(func)
    def wrapper(*args, **kargs) -> dict:
        materials = func(*args, **kargs)

        ## set tracking uri
        mlflow.set_tracking_uri('http://127.0.0.1:5000')

        ## experiment
        try:
            mlflow.create_experiment(
                name=materials['experiment_name'], 
                artifact_location='storage/.mlflow/.artifacts_store'
            )
            mlflow.set_experiment(
                experiment_name=materials['experiment_name']
            )
        except:
            mlflow.set_experiment(
                experiment_name=materials['experiment_name']
            )

        return materials
    
    return wrapper

# 
def log_model(func: Callable[[dict], dict]):
    @functools.wraps(func)
    def wrapper(*args, **kargs):
        materials = func(*args, **kargs)
        ## 
        val_predictions = materials['pipeline'].predict(materials['X_test'])
        val_fbeta = fbeta_score(
            y_true=materials['y_test'], y_pred=val_predictions, 
            beta=2
        )
        ##
        with mlflow.start_run():
            ###
            mlflow.log_params(
                params=materials['params']
            )
            ###
            mlflow.set_tags(
                {
                    'beta_f': 2,
                    'avg_fbeta': materials['avg_fbeta'],
                    'val_fbeta': val_fbeta
                }
            )
            ###
            mlflow.pyfunc.log_model(
                artifact_path='model', 
                python_model=MLflowModel(), 
                artifacts=materials['artifacts_path'], 
                signature=infer_signature(materials['X_train']), 
                pip_requirements=['joblib', 'mlxtend', 'imbalanced-learn', 'scikit-learn']
            )

        return materials
    
    return wrapper