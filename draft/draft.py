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

# class Onehot_Scaling_PCA for SFS
class FS_BaseUserDefinedTransformer(BaseEstimator, TransformerMixin):
    ##
    def __init__(self):
        ###
        scaler, pca = [('normalize', StandardScaler())], [('PCA', PCA(n_components=0.8))]
        steps = scaler + pca
        self.num_pro = Pipeline(steps)

        ###
        self.cat_pro = OneHotEncoder(drop='first', sparse_output=False)

    ##
    def _check_ndim(self, X: np.ndarray) -> tuple[np.ndarray, int]:
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
        for i in range(num_iters):
            try:
                X_[:1, i].astype(float)
                num_idxes.append(i)
            except:
                cat_idxes.append(i)
        
        return num_idxes, cat_idxes, X_

    def fit(self, X: np.ndarray, y=None):
        ###
        self.num_idxes, self.cat_idxes, X_ = self._category_detection(X)

        ### 
        if len(self.num_idxes) == 0:
            transformers = [('cat_pro', self.cat_pro, self.cat_idxes)]
        elif len(self.cat_idxes) == 0:
            transformers = [('num_pro', self.num_pro, self.num_idxes)]
        else:
            transformers = [('num_pro', self.num_pro, self.num_idxes), 
                           ('cat_pro', self.cat_pro, self.cat_idxes)]
            
        ###
        self.ct = ColumnTransformer(transformers=transformers, remainder='passthrough')
        self.ct.fit(X=X_)
        
        return self
    
    def transform(self, X: np.ndarray, y=None):
        X_, _ = self._check_ndim(X=X)
        return self.ct.transform(X_)
    

class SFS_OSP(FS_BaseUserDefinedTransformer):
    def fit(self, X: np.ndarray, y=None):
        ###
        self.num_idxes, self.cat_idxes, self.X_fit_ = self._category_detection(X)

        ###
        steps = self.scaling + self.factor_analysis
        self.num_pro = [('num_pro', Pipeline(steps), num_idxes)]

        self.cat_pro = [('cat_pro', self.ohe, cat_idxes)]

        ###
        transformers = self._get_transformers(cat_idxes, num_idxes)
        self.ct = ColumnTransformer(transformers, remainder='passthrough')
        self.ct.fit(X=self.X_fit_)

        return self

    def transform(self, X: np.ndarray, y=None):
        X_, _ = self._check_ndim(X=X)
        
        return self.ct.transform(X_)



