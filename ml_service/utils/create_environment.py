from azureml.core import Environment
from azureml.core.runconfig import DEFAULT_CPU_IMAGE,DEFAULT_GPU_IMAGE
from ml_service.utils.environment_variables import ENV
import os

def get_environment(workspace,
                    env_name,
                    conda_dependencies,
                    create_new=False,
                    enable_docker=None,
                    use_gpu=False):
    try:
        e = ENV()
        restore_env = None
        environments = Environment.list(workspace=workspace)
        for env in environments:
            if env == env_name:
                restore_env = environments[env_name]
        
        if restore_env is None or create_new:
            new_env = Environment.from_conda_specification(env_name,
                                                           os.path.join(e.source_train_directory,conda_dependencies))
            restore_env = new_env
            if enable_docker:
                restore_env.docker.enabled = enable_docker
                restore_env.docker.base_image = DEFAULT_GPU_IMAGE if use_gpu else DEFAULT_CPU_IMAGE
            restore_env.register(workspace)
        if restore_env is not None:
            print(f'created environment: {restore_env}')
        return restore_env
    except Exception as error:
        print(f'Founding error: {error} occurred')
        exit(1)
