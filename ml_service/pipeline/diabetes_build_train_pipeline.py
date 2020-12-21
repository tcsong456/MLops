from azureml.core import Workspace,Datastore,Dataset
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.core import Pipeline,PipelineData
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
                                  env_name=e.aml_env_name,
                                  conda_dependencies=e.aml_conda_train_dependent_files,
                                  create_new=e.rebuild_env)
    run_config = RunConfiguration()
    run_config.environment = environment
    
    if e.datastore_name:
        datastore_name = e.datastore_name
    else:
        datastore_name = aml_workspace.get_default_datastore().name
    run_config.environment.environment_variables['DATASTORE_NAME'] = datastore_name
    
    model_name_param = PipelineParameter(name='model_name',default_value=e.model_name)
    dataset_version_param = PipelineParameter(name='dataset_version',default_value=e.dataset_version)
    dataset_file_path = PipelineParameter(name='dataset_file_path',default_value='none')
    
    dataset_name = e.dataset_name
    if dataset_name not in aml_workspace.datasets:
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
        dataset.register(workspace=aml_workspace,
                         name=dataset_name,
                         description='registered dataset',
                         create_new_version=True,
                         tags={'format':'CSV'})
    
    pipeline_data = PipelineData('train',datastore=aml_workspace.get_default_datastore())
    train_step = PythonScriptStep(script_name=e.train_script_path,
                                  name='train_step',
                                  arguments=['--model-name',model_name_param,
                                             '--dataset-name',dataset_name,
                                             '--dataset-version',dataset_version_param,
                                             '--dataset-file-path',dataset_file_path,
                                             '--step-output',pipeline_data],
                                  compute_target=aml_compute,
                                  runconfig=run_config,
                                  source_directory=e.source_train_directory,
                                  outputs=[pipeline_data],
                                  allow_reuse=False)
    print('Train step created!')    
    
    eval_step = PythonScriptStep(name='eval_step',
                                 script_name=e.eval_script_path,
                                 compute_target=aml_compute,
                                 source_directory=e.source_train_directory,
                                 arguments=['--model-name',model_name_param,
                                            '--allow-run-cancel',e.allow_run_cancel],
                                 runconfig=run_config,
                                 allow_reuse=False)
    print('EVAL step created!')
    
    register_step = PythonScriptStep(name='register_step',
                                     script_name=e.register_script_path,
                                     compute_target=aml_compute,
                                     source_directory=e.source_train_directory,
                                     inputs=[pipeline_data],
                                     arguments=['--model-name',model_name_param,
                                                '--step-input',pipeline_data],
                                     runconfig=run_config,
                                     allow_reuse=False)
    print('Register step created!')
    
    if e.run_evaluation:
        print('evaluation step is included')
        eval_step.run_after(train_step)
        register_step.run_after(eval_step)
        steps = [train_step,eval_step,register_step]
    else:
        print('evaluation step is excluded')
        register_step.run_after(train_step)
        steps = [train_step,register_step]
        
    train_pipeline = Pipeline(workspace=aml_workspace,
                              steps=steps)
    train_pipeline.validate()
    published_pipeline = train_pipeline.publish(name=e.pipeline_name,
                                                description='model training pipeline',
                                                version=e.build_id)
    print(f'{published_pipeline.name} is published1')

if __name__ == '__main__':
    main()


    