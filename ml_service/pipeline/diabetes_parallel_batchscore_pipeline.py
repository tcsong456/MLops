from azureml.data.datapath import DataPath
from azureml.core import Workspace,Datastore,Dataset
from azureml.pipeline.core import Pipeline,PipelineData,RunConfiguration,Pipelineparameter
from azureml.pipeline.steps import ParallelRunConfig,ParallelRunStep,PythonScriptStep
from ml_service.pipeline.load_data import create_sample_data_csv
from ml_service.utils.create_environment import get_environment
from ml_service.utils.create_compute import get_compute
from ml_service.utils.environment_variables import ENV
import os

def get_or_create_datastore(datastore_name,
                            env,
                            workspace,
                            input=True):
    if datastore_name is not None:
        raise ValueError('datastore name can not be empty!')
    
    container_name = env.scoring_datastore_input_container if input else env.scoring_datastore_output_container
    if datastore_name in workspace.datastores:
        datastore = workspace.datastores[datastore_name]
    elif container_name is not None and env.scoring_datastore_access_key is not None:
        datastore = Datastore.register_azure_blob_container(workspace=workspace,
                                                            datastore_name=datastore_name,
                                                            container_name=container_name,
                                                            account_name=env.scoring_datastore_storage_name,
                                                            account_key=env.scoring_datastore_access_key)
    else:
        raise ValueError('no datastore_name exsting in current workspace nor enough info provided to build a new datastore')
    
    return datastore

def get_input_dataset(workspace,
                      datastore,
                      env):
    scoring_input_ds = Dataset.Tabuar.from_delimited_files(path=DataPath(datastore,env.scoring_datastore_input_filename))
    scoring_input_ds = scoring_input_ds.register(workspace=workspace,
                                                 name=env.scoring_dataset_name,
                                                 tag={'purpose':'for scoring','format':'csv'},
                                                 create_new_version=True).as_named_input(env.scoring_dataset_name)
    
    return scoring_input_ds

def get_fallback_input_dataset(workspace,
                               env):
    create_sample_data_csv(env.scoring_datastore_input_filename,for_scoring=True)
    if not os.path.exists(env.scoring_datastore_input_filename):
        raise FileNotFoundError(f'dataset: {env.scoring_datastore_input_filename} not found in current direcotry!')
    
    datastore = workspace.get_default_datastore()
    scoring_input_ds = datastore.upload_files(files=[env.scoring_datastore_input_filename],
                                              target_path='scoring_input',
                                              overwrite=False)
    dataset = Dataset.Tabular.from_delimited_files(scoring_input_ds).register(workspace=workspace,
                                                                              name=env.scoring_dataset_name,
                                                                              create_new_version=False).as_named_input(env.scoring_dataset_name)
    
    return dataset

def get_output_location(workspace,
                        output_datastore=None):
    if output_datastore is not None:
        output_loc = PipelineData(name=output_datastore.name,
                                  datastore=output_datastore)
    else:
        output_loc = PipelineData(name='output_loc',
                                  datastore=workspace.get_default_datastore())
    
    return output_loc

def get_inputds_outputloc(env,
                          workspace):
    if env.scoring_datastore_storage_name is None:
        input_ds = get_fallback_input_dataset(workspace=workspace,
                                              env=env)
        output_loc = get_output_location(workspace=workspace)
    else:
        input_datastore = get_or_create_datastore(f'{env.scoring_datastore_storage_name}_in',
                                                  env=env,
                                                  workspace=workspace,
                                                  input=True)
        output_datastore = get_or_create_datastore(f'{env.scoring_datastore_storage_name}_out',
                                                   env=env,
                                                   workspace=workspace,
                                                   input=False)
        input_ds = get_input_dataset(workspace=workspace,
                                     datastore=input_datastore,
                                     env=env)
        output_loc = get_output_location(workspace=workspace,
                                         output_datastore=output_datastore)
    
    return input_ds,output_loc

def get_run_configs(workspace,
                    compute_target,
                    env):
    environment = get_environment(workspace=workspace,
                                  env_name=env.aml_env_scoring_name,
                                  conda_dependencies=env.aml_conda_score_file,
                                  create_new=env.aml_rebuild_scoring_env,
                                  enable_docker=True,
                                  use_gpu=env.use_gpu_for_scoring)
    score_run_config = ParallelRunConfig(environment=environment,
                                         entry_script=env.batchscore_script_path,
                                         source_directory=env.source_train_directory,
                                         error_threshold=10,
                                         output_action='append_row',
                                         compute_target=compute_target,
                                         node_count=env.max_scoring_nodes,
                                         run_invocation_timeout=300)
    copy_run_config = RunConfiguration()
    copy_run_config.environment = get_environment(workspace=workspace,
                                                  env_name=env.aml_env_scorecopy_name,
                                                  conda_denpendencies=env.aml_conda_score_file,
                                                  create_new=env.rebuild_scoring_env,
                                                  enable_docker=True,
                                                  use_gpu=env.use_gpu_for_scoring)
    
    return score_run_config,copy_run_config

def get_scoring_pipeline(score_run_config,
                         copy_run_config,
                         scoring_input,
                         output_loc,
                         env,
                         compute_target,
                         workspace):
    model_name_param = Pipelineparameter('model-name',default_value='')
    model_version_param = Pipelineparameter('model-version',default_value='')
    model_tag_name_param = Pipelineparameter('model-tag-name',default_value='')
    model_tag_value_param = Pipelineparameter('model-tag_value',deffault_value='')
    
    score_step = ParallelRunStep(name='batch_scoring',
                                 inputs=[scoring_input],
                                 output=output_loc,
                                 arguments=['--model-name',model_name_param,
                                            '--model-version',model_version_param,
                                            '--model-tag-name',model_tag_name_param,
                                            '--model-tag-value',model_tag_value_param],
                                allow_resue=False,
                                parallel_run_config=score_run_config,
                                source_directory=env.source_train_directory,
                                )
    copy_step = PythonScriptStep(script_name=env.batch_scorecopy_script_path,
                                 source_directory=env.source_train_directory,
                                 inputs=[output_loc],
                                 allow_reuse=False,
                                 runconfig=copy_run_config,
                                 compute_target=compute_target,
                                 arguments=['--output_path',output_loc,
                                            '--scoring_datastore',
                                            env.scoring_datastore_storage_name if env.scoring_datastore_storage_name is not None else '',
                                            '--score_container',
                                            env.scoring_datastore_output_container if env.scoring_datastore_output_container is not None else '',
                                            '--scoring_datastore_key',
                                            env.scoring_datastore_access_key if env.scoring_datastore_access_key is not None else '',
                                            '--scoring_output_filename',
                                            env.scoring_datastore_output_filename if env.scoring_datastore_output_filename is not None else '']
                                 )
    
    steps = [score_step,copy_step]
    
    return Pipeline(workspace=workspace,steps=steps)

def build_batchscore_pipeline():
    try:
        env = ENV()
        ws = Workspace.get(name=env,
                           subscription_id=env.subscription_id,
                           resource_group=env.resource_group)
        compute_target = get_compute(workspace=ws,
                                     compute_name=env.compute_scoring_name,
                                     vm_size=env.vm_priority_scoring,
                                     for_batch_scoring=True)
        score_run_config,copy_run_config = get_run_configs(workspace=ws,
                                                           env=env,
                                                           compute_target=compute_target)
        scoring_input,output_loc = get_inputds_outputloc(env=env,
                                                         workspace=ws)
        scoring_pipeline = get_scoring_pipeline(score_run_config=score_run_config,
                                                copy_run_config=copy_run_config,
                                                scoring_input=scoring_input,
                                                output_loc=output_loc,
                                                env=env,
                                                compute_target=compute_target,
                                                workspace=ws)
        
        published_pipeline = scoring_pipeline.publish(name=env.scoring_pipeline_name,
                                                      description="Diabetes Batch Scoring Pipeline")
        pipeline_id = f'##vso[task.setvariable variable=pipeline_id;isOutput=true]{published_pipeline.id}'
        print(pipeline_id)
    except Exception as ex:
        print(ex)
    
    
    
    

