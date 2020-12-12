from azureml.core import Datastore,Dataset
from azureml.core.run import Run
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import argparse
import pprint
import json
import os
import joblib

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

def split_data(df):
    X = df.drop('Y',axis=1).values
    Y = df['Y']
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=5)
    return {'train':{'X':x_train,'Y':y_train},
            'test':{'X':x_test,'Y':y_test}}

def train_model(data,ridge_args):
    ridge_model = Ridge(**ridge_args)
    ridge_model.fit(data['train']['X'],data['train']['Y'])
    return ridge_model

def get_model_metrics(model,data):
    preds = model.predict(data['test']['X'])
    mse = mean_squared_error(data['test']['Y'],preds)
    metric = {"mse":mse}
    return metric

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
    
    with open('parameters.json') as f:
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
        if dataset_file_path == 'none':
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
    
    run.input_datasets['training_data'] = dataset
    run.parent.tag('dataset_id',dataset.id)
    
    df = dataset.to_pandas_dataframe()
    data = split_data(df)
    model = train_model(data,train_args)
    metrics = get_model_metrics(model,data)
    for key,value in metrics.items():
        run.log(key,value)
        run.parent.log(key,value)
    
    os.makedirs(step_output,exist_ok=True)
    model_path = os.path.join(step_output,model_name)
    joblib.dump(value=model,filename=model_path)
    
    os.makedirs('outputs',exist_ok=True)
    output_path = os.path.join('outputs',model_name)
    joblib.dump(value=model,filename=output_path)
    
    run.tag('run_type',value='train')
    run.complete()

if __name__ == '__main__':
    main()

