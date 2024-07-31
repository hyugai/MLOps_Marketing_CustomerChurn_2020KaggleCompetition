import os, sys, joblib
cwd = os.getcwd()
os.chdir('../')
modules_path = os.getcwd()
if modules_path not in sys.path:
    sys.path.append(modules_path)
os.chdir(cwd)

from usr_modules.notebook.features_engineering import SFS_OSP

sfs = joblib.load('../notebooks/.artifacts/ohe_quantiletransform.joblib')