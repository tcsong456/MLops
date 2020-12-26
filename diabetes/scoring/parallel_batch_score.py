import sys
import joblib
import numpy as np
import pandas as pd
from utils.model_utils import get_model
from azureml.core import Model

def parse_args():
    model_name_param = [sys.argv[idx+1] for idx,item in enumerate(sys.argv) if item == '--model-name']
    if len(model_name_param) == 0:
        raise ValueError('model name must be provided')
    model_name = model_name_param[0]
    
    model_version_param = [sys.argv[idx+1] for idx,item in enumerate(sys.argv) if item == '--model-version']
    model_version = None if len(model_version_param) == 0 or len(model_version_param[0].strip()) == 0 else model_version_param[0]
    
    model_tag_name_param = [sys.argv[idx+1] for idx,item in enumerate(sys.argv) if item == '--model-tag-name']
    model_tag_name = None if len(model_tag_name_param) == 0 or len(model_tag_name_param[0].strip()) == 0 else model_tag_name_param[0]
    
    model_tag_value_param = [sys.argv[idx+1] for idx,item in enumerate(sys.argv) if item == '--model-tag-value']
    model_tag_version = None if len(model_tag_value_param) == 0 or len(model_tag_value_param[0].strip()) == 0 else model_tag_value_param[0]
    
    return [model_name,model_version,model_tag_name,model_tag_version]

def init():
    try:
        print('Loading Model')
        model_params = parse_args()
        aml_model = get_model(model_name=model_params[0],
                              model_version=model_params[1],
                              tag_name=model_params[2],
                              tag_value=model_params[3])
        global model
        model_path = Model.get_model_path(model_name=aml_model.name,
                                          version=aml_model.version)
        model = joblib.load(model_path)
        print(f'model:{aml_model.name} downloading is successful')
    except Exception as ex:
        print(ex)

def run(mini_batch):
    try:
        result = None
        for _,row in mini_batch.iterrows():
            pred = model.predict(row.values.reshape(1,-1))
        result = (np.array(pred) if result is None else np.vstack([result,pred]))
        
        return ([] if result is None else mini_batch.join(pd.DataFrame(result,columns=['score'])))
    except Exception as ex:
        print(ex)
