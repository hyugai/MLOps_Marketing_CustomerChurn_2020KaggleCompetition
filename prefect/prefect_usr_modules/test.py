import os, sys

cwd = os.getcwd()
os.chdir('../../')
modules_dir = os.getcwd()
os.chdir(cwd)
if modules_dir not in sys.path:
    sys.path.append(modules_dir)

from usr_modules.mlflow_usr_defined import MLflowModel
