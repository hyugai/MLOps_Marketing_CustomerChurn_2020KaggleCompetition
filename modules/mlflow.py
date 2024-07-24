# mlflow
from mlflow.pyfunc import PythonModel

# others
import joblib

## customized model for mlflow
class MLflowModel(PythonModel):
    def __init__(self) -> None:
        self.model = None
        self.FeatureSelector = None    

    def load_context(self, context):
        self.model = joblib.load(context.artifacts['model'])
        self.FeatureSelector = joblib.load(context.artifacts['feature_selector'])

    def predict(self, context, model_input, params=None):
        params = params or {'predict_method': 'predict'}
        predict_method = params.get('predict_method')

        selected_model_input = self.FeatureSelector.transform(model_input)

        if predict_method == 'predict':
            return self.model.predict(selected_model_input)
        elif predict_method == 'predict_proba':
            return self.model.predict_proba(selected_model_input)
        elif predict_method == 'predict_log_proba':
            return self.model.predict_log_proba(selected_model_input)
        else:
            raise ValueError(f'The predict method \'{predict_method}\' is not supported')
        
