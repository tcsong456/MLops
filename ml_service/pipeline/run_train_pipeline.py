import argparse
from ml_service.utils.environment_variables import ENV
from azureml.core import Workspace,Experiment
from azureml.pipeline.core import PublishedPipeline

def main():
    parser = argparse.ArgumentParser('trigger_pipeline')
    arg = parser.add_argument
    arg('--output-write-file',type=str,default='pipeline_id_recorder',
        help='the text file to write piepeline id')
    arg('--skip-train-exc',action='store_true',
        help='option to skip train excecution')
    args = parser.parse_args()
    
    e = ENV()
    print(e.build_id)
    aml_workspace = Workspace.get(name=e.workspace_name,
                                  subscription_id=e.subscription_id,
                                  resource_group=e.resource_group
                                  ) 
    
    published_pipeline = PublishedPipeline.list(aml_workspace)
    matched_pipes = []
    for pipe in published_pipeline:
        if pipe.name == e.pipeline_name:
            if pipe.version == e.build_id:
                matched_pipes.append(pipe)
    
    if len(matched_pipes) > 1:
        published_pipeline = None
        raise Exception('there should be only one published pipeline')
    elif len(matched_pipes) == 0:
        published_pipeline = None
        raise Exception('no pipeline is published on the provided workspace!')
    else:
        published_pipeline = matched_pipes[0]
        print(f'published piepeline id is {published_pipeline.id}')
    
        if args.output_write_file is not None:
            with open(args.output_write_file,'w') as output_file:
                output_file.write(published_pipeline.id)
            
        if args.skip_train_exc is False:
            pipeline_param = {'model_name':e.model_name}
            tags = {'build_id':e.build_id}
            if e.build_uri is not None:
                tags = {'build_uri':e.build_uri}
            exp = Experiment(workspace=aml_workspace,
                             name=e.experiment_name)
            run = exp.submit(published_pipeline,
                             tags=tags,
                             pipeline_param=pipeline_param)
            print(f'pipeline {published_pipeline.id} initiated,run id: {run.id}')

if __name__ == '__main__':
    main()
            