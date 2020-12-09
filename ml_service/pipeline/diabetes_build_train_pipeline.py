from utils.env_variables import Env
from azureml.core import Workspace,Datastore,Dataset
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core.graph import PipelineParameter
from ml_service.utils.create_compute import get_compute
from ml_service.utils.create_environment import get_environment
from ml_service.utils.environment_variables import ENV
from ml_service.pipeline.load_data import create_sample_data_csv
import os

def main():
    e = ENV()
    aml_workspace = Workspace.get(
                                  name=e.workspace_name,
                                  subscription_id=e.subscription_id,
                                  resource_group=e.resource_group
                                  )
    print(f'workspace:{aml_workspace}')
    
    aml_compute = get_compute(workspace=aml_workspace,
                              compute_name=e.compute_name,
                              vm_size=e.vm_size)
    if aml_compute is not None:
        print(f'compute target: {aml_compute} is created')
    
    environment = get_environment(workspace=aml_workspace,
                                  env_name=e.env_name,
                                  conda_dependencies=e.aml_conda_train_dependent_files,
                                  create_new=e.rebuild_env)
    run_config = RunConfiguration()
    run_config.environment = environment
    
    if e.datastore_name:
        datastore_name = e.datastore_name
    else:
        datastore_name = aml_workspace.get_default_datastore().name
    run_config.environment.environmentVariables['DATASTORE_NAME'] = datastore_name
    
    model_name_param = PipelineParameter(name='model_name',default_value=e.model_name)
    dataset_version_param = PipelineParameter(name='dataset_version',default_value=e.dataset_version)
    dataset_file_path = PipelineParameter(name='dataset_file_path',default_value='none')
    
    if e.dataset_name not in aml_workspace.datasets:
        create_sample_data_csv()
        file_name = 'diabetes.csv'
        if not os.path.exists(file_name):
            raise Exception(f'{file_name} does not exist!')
        datastore = Datastore.get(aml_workspace,datastore_name)
        target_path = 'training/'
        datastore.upload_files(files=[file_name],
                               target_path=target_path,
                               overwrite=True,
                               show_progress=True)
        path_on_datastore = os.path.join(target_path,file_name)
        dataset = Dataset.Tabular.from_delimited_files(path=(datastore,path_on_datastore))
        
        #%%
from azureml.core.runconfig import RunConfiguration
run_config = RunConfiguration()
run_config
    