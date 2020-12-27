import argparse
from azure.storage.blob import ContainerClient
from azureml.pipeline.core import PublishedPipeline
from azureml.core import Workspace,Experiment
from ml_service.utils.environment_variables import ENV

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_id',type=str,
                        help='the id of the published pipeoline to download')
    args = parser.parse_args()
    
    return args

def get_pipeline(workspace,
                 env,
                 pipeline_id=None):
    if pipeline_id is not None:
        scoring_pipeline = PublishedPipeline.get(workspace,id=pipeline_id)
    else:
        pipeline_list = PublishedPipeline.list(workspace)
        scoring_pipeline = [pl for pl in pipeline_list if pl.name == env.scoring_pipeline_name]
        if len(scoring_pipeline) == 0:
            raise ValueError('no available pipeline to download!')
        else:
            scoring_pipeline = scoring_pipeline[0]
    
    return scoring_pipeline

def copy_output(step_id,
                env):
    print(step_id)
    account_url = f'https://{env.scoring_datastore_storage_name}.blob.core.windows.net'
    src_blob_name = f'azureml/{step_id}/{env.scoring_datastore_storage_name}_out/parallel_run_step.txt'
    src_blob_url = f'{account_url}/{env.scoring_datastore_output_container}/{src_blob_name}'
    container_client = ContainerClient(account_url=account_url,
                                       container_name=env.scoring_datastore_output_container,
                                       credential=env.scoring_datastore_access_key)
    src_blob_properties = container_client.get_blob_client(src_blob_name).get_blob_properties()
    
    destfolder = src_blob_properties.last_modified.date().isoformat()
    file_time = (src_blob_properties.last_modified.time()).isoformat('miliseconds').replace(':','_').replace('.','_')
    filename_parts = env.scoring_datastore_output_filename.split('.')
    dest_blob_name = f'{destfolder}/{filename_parts[0]}_{file_time}.{filename_parts[1]}'
    dest_client = container_client.get_blob_client(dest_blob_name)
    dest_client.start_copy_from_url(src_blob_url)

def run_batchscore_pipeline():
    try:
        env = ENV()
#        args = parse_args()
        pipeline_id = env.pipeline_id
        workspace = Workspace.get(name=env.workspace_name,
                                  subscription_id=env.subscription_id,
                                  resource_group=env.resource_group)
        
        ds = workspace.get_default_datastore()
        print(ds,ds.name,ds.account_name,ds.container_name)
        
        scoring_pipeline = get_pipeline(workspace=workspace,
                                        env=env,
                                        pipeline_id=pipeline_id)
        exp = Experiment(workspace,name=env.experiment_name)
        run = exp.submit(scoring_pipeline,
                         pipeline_parameters={
                                            'model-name':env.model_name_scoring,
                                            'model-version':env.model_version_scoring,
                                            'model-tag-name':" ",
                                            'model-tag-value':" "})
        run.wait_for_completion(show_output=True)
        if run.get_status() == 'Finished':
            print(run.get_steps())
            copy_output(list(run.get_steps())[0].id,env)
        print('running scccessful!')
    except Exception as e:
        import traceback
        traceback.print_exc(limit=20)
        exit(1)

if __name__ == '__main__':
    run_batchscore_pipeline()
    
#https://amlworkspace5060042570.blob.core.windows.net/azureml-blobstore-c564fdde-7e8c-4046-a73b-be96a576bb57/
#azureml/30a88d34-c18b-460a-aa9a-d5833c37330f/output_loc/parallel_run_step.txt
    