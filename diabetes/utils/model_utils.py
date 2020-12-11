from azureml.core import Run
from azureml.core.model import Model

def get_current_workspace():
    run = Run.get_context(allow_offline=False)
    exp = run.experiment
    return exp.workspace

def get_model(
              model_name,
              model_version=None,
              workspace=None,
              tag_name=None,
              tag_value=None
                              ):
    if workspace is None:
        print('no workspace is provided,will get the current available workspace')
        workspace = get_current_workspace()
    
    tags = None
    if tag_name is not None or tag_value is not None:
        if tag_name is None or tag_value is None:
            raise ValueError('both tag_name and tag_value must be provided with a value')
        tags = [[tag_name,tag_value]]
    
    model = None
    if model_version is not None:
        model = Model(workspace=workspace,
                      name=model_name,
                      tags=tags,
                      version=model_version)
    else:
        models = Model.list(workspace=workspace,
                            name=model_name,
                            tags=tags,
                            latest=True)
        if len(models) == 1:
            model = models[0]
        if len(models) > 1:
            raise Exception('only one model expected!')
        
    return model
