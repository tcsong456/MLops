from utils.env_variables import Env
from azureml.core import Workspace,Datastore,Dataset

def main():
    e = Env()
    aml_workspace = Workspace.get(
                                  name=e.workspace_name,
                                  subscription_id=e.subscription_id,
                                  resource_group=e.resource_group
                                  )
    