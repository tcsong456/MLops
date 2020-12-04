from azureml.core import Datastore,Workspace,Dataset
from azureml.core.run import Run
import argparse
import pprint
import json
import os

def register_dataset(workspace,
                     datastore_name,
                     dataset_name,
                     file_path):
    datastore = Datastore.get(workspace=workspace,
                              datastore_name=datastore_name)
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore,file_path))
    dataset = dataset.regisetr(workspace=workspace,
                               name=dataset_name,
                               create_new_version=True)#either create new version if existed of exist_ok=True
    
    return dataset

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model-name',type=str,default='diabetes_model.pkl',
        help='the name of the model')
    arg('--dataset-name',type=str,
         help='dataset_name is always required to get the exact version \
         of dataset wanted instead of the one during pipeline creation')
    arg('--step-output',type=str,
        help='output of data passing to next step')
    arg('--dataset-version',type=str,
        help='wanted dataset version')
    arg('--dataset-file-path',type=str,
        help='new dataset to register')
    args = parser.parse_args()
    arg_dict = {'model_name':args.model_name,
                'datset_name':args.dataset_name,
                'step_output':args.step_output,
                'dataset_version':args.dataset_version}
    pprint.pprint(arg_dict)
    
    model_name = args.model_name
    dataset_name = args.dataset_name
    step_output = args.step_output
    dataset_file_path = args.dataset_file_path
    dataset_version = args.dataset_version
    run = Run.get_context()
    
    with open('diabetes/parameters.json') as f:
        pars = json.load(f)
    try:
        train_args = pars['training']
    except KeyError:
        print('training key is not found!')
        train_args = {}
    print(f'training params:{train_args}')
    for key,value in train_args.items():
        run.log(key,value)
        run.parent.log(key,value)
    
    if dataset_name:
        if dataset_file_path:
            dataset = Dataset.get_by_name(workspace=run.experiment.workspace,
                                          name=dataset_name,
                                          version=dataset_version)
        else:
            dataset = register_dataset(workspace=run.experiment.workspace,
                                       datastore_name=os.environ.get('DATASTORE_NAME'),
                                       dataset_name=dataset_name,
                                       file_path=dataset_file_path)
    else:
        raise Exception('No dataset is provided')
    
    run.input_datasets['training'] = dataset

#%%

    
#parser = argparse.ArgumentParser()
#arg = parser.add_argument
#arg('--model-name',type=str,default='diabetes_model.pkl',
#    help='the name of the model')
#arg('--dataset-name',type=str,
#     help='dataset_name is always required to get the exact version \
#     of dataset wanted instead of the one during pipeline creation')
#arg('--step-output',type=str,
#    help='output of data passing to next step')
#arg('--dataset-version',type=str,
#    help='wanted dataset version')
#arg('--dataset-file-path',type=str,
#    help='new dataset to register')
#args = parser.parse_args()
#print(args.dataset_file_path)

#print(arg_dict)