from azureml.core.model import Model
from azureml.core import Dataset,Run
import argparse
import json
import os
import joblib
import traceback
import sys

def model_already_registered(model_name,run_id,exp):
    model_list = Model(workspace=exp.workspace,
                       run_id=run_id,
                       name=model_name)
    if len(model_list) >= 1:
        error = f'model: {model_name} with run_id: {run_id}  has already been registered at workspace: {exp.workspace}'
        print(error)
        raise Exception(error)
    else:
        print('model is not registered for this workspace')

def register_aml_model(run_id,
                       exp,
                       model_tags,
                       model_name,
                       model_path,
                       dataset_id,
                       build_id=None,
                       build_uri=None):
    try:
        tags_value = {'area':'diabetes_regression',
                      'run_id':run_id,
                      'experiment_name':exp.name}
        tags_value.update(model_tags)
        if build_id is not None:
            model_already_registered(model_name,run_id,exp)
            tags_value['BuildId'] = build_id
            if build_uri is not None:
                tags_value['BuildUri'] = build_uri
        
        model = Model.register(workspace=exp.workspace,
                               model_path=model_path,
                               tags=tags_value,
                               model_name=model_name,
                               datasets=[('training_data',Dataset.get_by_id(exp.workspace,dataset_id))])
        print(f'{model_name} has been registered,\nmodel description: {model.description},\nmodel version: {model.version}')
    except Exception:
        traceback.print_exc(limit=None,file=None,chain=True)
        print('model registration failed!')
        raise
        
def main():
    run = Run.get_context()
    exp = run.experiment
    run_id = 'amlcompute'
    
    parser = argparse.ArgumentParser('registeration')
    arg = parser.add_argument
    arg('--model-name',type=str,default='diabetes_regression_model.pkl',
        help='the name of the model')
    arg('--step-input',type=str,
        help='the input path of model')
    arg('--run-id',type=str,
        help='Training run id')
    args = parser.parse_args()
    
    if args.run_id is not None:
        run_id = args.run_id
    if run_id == 'amlcompute':
        run_id = run.parent.id
    model_name = args.model_name
    model_path = args.step_input
    
    with open('parameters.json') as param:
        par = json.load(param)
    try:
        register_tags = par['registration']
    except KeyError:
        register_tags = {'tags':[]}
    
    model_tags = {}
    try:
        for tag in register_tags:
            mtag = run.parent.get_metrics()[tag]
            model_tags[tag] = mtag
    except KeyError:
        print('could not find tag in parent run')
    
    model_file = os.path.join(model_path,model_name)
    model = joblib.load(model_file)
    parent_tags = run.parent.get_tags()
    
    try:
        build_id = parent_tags['BuildId']
    except KeyError:
        build_id = None
        print('BuildId not exist in parent_tags')
        print(f'parent tags: {parent_tags}')
    try:
        build_uri = parent_tags['BuildUri']
    except KeyError:
        build_uri = None
        print('BuildUri not exit in parent_tags')
        print(f'parent_tags: {parent_tags}')
    
    if model is not None:
        dataset_id = parent_tags['dataset_id']
        if build_id is None:
            register_aml_model(run_id=run_id,
                               exp=exp,
                               model_tags=model_tags,
                               model_name=model_name,
                               model_path=model_file,
                               dataset_id=dataset_id)
        elif build_uri is None:
            register_aml_model(run_id=run_id,
                               exp=exp,
                               model_tags=model_tags,
                               model_name=model_name,
                               model_path=model_file,
                               dataset_id=dataset_id,
                               build_id=build_id)
        else:
            register_aml_model(run_id=run_id,
                               exp=exp,
                               model_tags=model_tags,
                               model_name=model_name,
                               model_path=model_file,
                               dataset_id=dataset_id,
                               build_id=build_id,
                               build_uri=build_uri)
    else:
        print('Model not founding,skipping model registration!')
        sys.exit(0)

if __name__ == '__main__':
    main()
    
        