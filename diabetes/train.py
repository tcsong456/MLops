from azureml.core import Datastore,Workspace,Dataset

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

#%%
import argparse
import pprint
parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--model-name',type=str,default='diabetes_model.pkl',
    help='the name of the model')
arg('--dataset-name',type=str,
     help='dataset_name is always required to get the exact version \
     of dataset wanted instead of the one during pipeline creation')