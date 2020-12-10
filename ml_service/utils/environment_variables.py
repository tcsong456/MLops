from dataclasses import dataclass
import os
from typing import Optional

@dataclass(frozen=True)
class ENV:
    workspace_name: Optional[str] = os.environ.get('WORKSPACE_NAME')
    subscription_id: Optional[str] = os.environ.get('SUBSCRIPTION_ID')
    resource_group: Optional[str] = os.environ.get('RESOURCE_GROUP')
    vm_priority: Optional[str] = os.environ.get('AML_CLUSTER_PRIORITY','lowpriority')
    vm_priority_scoring: Optional[str] = os.environ.get('AML_CLUSTER_PRIORITY_SCORING','lowpriority')
    vm_size: Optional[str] = os.environ.get('AML_COMPUTE_CLUSTER_CPU_SKU')
    min_nodes: Optional[int] = int(os.environ.get('AML_CLUSTER_MIN_NODES',0))
    min_nodes_scoring: Optional[int] = int(os.environ.get('AML_CLUSTER_MIN_NODES_SCORING',0))
    max_nodes: Optional[int] = int(os.environ.get('AML_CLUSTER_MAX_NODES',4))
    max_nodes_scoring: Optional[int] = int(os.environ.get('AML_CLUSTER_MAX_NODES_SCORING',4))
    source_train_directory: Optional[str] = os.environ.get('SOURCE_TRAIN_DIRECTORY','diabetes')
    aml_conda_train_dependent_files: Optional[str] = os.environ.get('AML_CONDA_TRAIN_DEPENDENT_FILES','conda_dependencies.yml')
    aml_env_name: Optional[str] = os.environ.get('AML_ENV_NAME')
    rebuild_env: Optional[bool] = os.environ.get('AML_REBUILD_ENVIRONMENT')
    model_name: Optional[str] = os.environ.get('MODEL_NAME')
    dataset_name: Optional[str] = os.environ.get('DATASET_NAME')
    build_id: Optional[str] = os.environ.get('BUILD_BUILDID')
    pipeline_name: Optional[str] = os.environ.get('TRAINING_PIPELINE_NAME')
    compute_name: Optional[str] = os.environ.get('AML_COMPUTE_CLUSTER_NAME')
    datastore_name: Optional[str] = os.environ.get('DATASTORE_NAME')
    dataset_version: Optional[str] = os.environ.get('DATASET_VERSION')
    train_script_path: Optional[str] = os.environ.get('TRAIN_SCRIPT_PATH')