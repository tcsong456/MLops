from azureml.pipeline.core import PublishedPipeline
from azureml.core import Workspace,Experiment
from ml_service.utils.environment_variables import ENV
import os

def main():
    e = ENV()
    aml_workspace = Workspace.get(name='aml-workspace',
                                  subscription_id=e.subscription_id,
                                  resource_group=e.resource_group)
    
    publish_pipeline_id = os.environ.get('AMLPIPELINE_ID')
    if publish_pipeline_id is not None:
        published_pipeline = PublishedPipeline.get(workspace=aml_workspace,
                                                   id=publish_pipeline_id)
    else:
        raise Exception('you need to save the id of a published_pipeline first')
    
    exp = Experiment(workspace=aml_workspace,
                     name=e.experiment_name)
    pipeline_parameters = {'model_name':e.model_name}
    tags = {'BuildId':e.build_id,
            'BuildUri':e.build_uri}
    run = exp.submit(published_pipeline,
                     tags=tags,
                     pipeline_parameters=pipeline_parameters)
    print('pipeline run is initiated,run_id: {run.id}')

if __name__ == '__main__':
    main()

