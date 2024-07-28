# %% "LOAD LIBRARIES"
# data structures
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
## settings
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_theme('notebook')

# models selection
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score

# metrics
from sklearn.metrics import fbeta_score, make_scorer

# pipeline
from imblearn.pipeline import Pipeline

# compose
from sklearn.compose import ColumnTransformer

# preprocessings
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder
from category_encoders.cat_boost import CatBoostEncoder

# decomposition
from sklearn.decomposition import PCA

# features selection
from mlxtend.feature_selection import SequentialFeatureSelector

# resamplings
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

# algorithms
## linear_model
from sklearn.linear_model import LogisticRegression
## neighbors
from sklearn.neighbors import KNeighborsClassifier
## svm
from sklearn.svm import SVC
## tree
from sklearn.tree import DecisionTreeClassifier
## ensample
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# others
import re
from sklearn.base import BaseEstimator, TransformerMixin

# class SFS Base
# %% "FS_BaseUserDefinedTransformer"
class FS_BaseUserDefinedTransformer(BaseEstimator, TransformerMixin):
    ## 
    def __init__(self, 
                 ohe=None, other_encoders=None,
                 scaling: list=[], factor_analysis: list=[]) -> None:
        ###
        self.scaling, self.factor_analysis = scaling, factor_analysis
        ### 
        self.ohe, self.other_encoders = ohe, other_encoders


    ##
    def _check_ndim(self, X: np.ndarray) -> tuple[np.ndarray, int]:
        ###
        if X.ndim == 2:
            X_ = X
            num_iters = X.shape[1]
        else:
            X_ = X.reshape(-1, 1)
            num_iters = 1

        return X_, num_iters
    
    ##
    def _category_detection(self, X: np.ndarray) -> tuple[list, list, np.ndarray]:
        ###
        num_idxes, cat_idxes = [], []

        ### check dimension
        X_, num_iters = self._check_ndim(X=X)

        ###
        for i in range(num_iters):
            try:
                X_[:1, i].astype(float)
                num_idxes.append(i)
            except:
                cat_idxes.append(i)
        
        return num_idxes, cat_idxes, X_
    
    ##
    def _get_transformers(self, cat_idxes: list=[], num_idxes: list=[]) -> list:
        if len(num_idxes) == 0:
            transformers = self.cat_pro
        elif len(cat_idxes) == 0:
            transformers = self.num_pro
        else:
            transformers = self.num_pro + self.cat_pro

        return transformers 
    
    ## 
    def fit(self, X: np.ndarray, y=None):
        ###
        transformers = self._get_transformers(self.cat_idxes, self.num_idxes)
        self.ct = ColumnTransformer(transformers, remainder='passthrough')
        self.ct.fit(self.X_fit_)

        return self
    
    ##
    def transform(self, X: np.ndarray, y=None):
        X_, _ = self._check_ndim(X=X)

        return self.ct.transform(X=X_)
    

# %% "SFS_OSP"
class SFS_OSP(FS_BaseUserDefinedTransformer): # Onehot_Scaling_Pca
    def fit(self, X: np.ndarray, y=None):
        ###
        self.num_idxes, self.cat_idxes, self.X_fit_ = self._category_detection(X)

        ###
        steps = self.scaling + self.factor_analysis
        self.num_pro = [('num_pro', Pipeline(steps), self.num_idxes)]

        self.cat_pro = [('cat_pro', self.ohe, self.cat_idxes)]

        ###
        super().fit(X=X)
        
        return self
    
# %%