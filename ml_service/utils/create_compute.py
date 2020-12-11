from azureml.core.compute import AmlCompute,ComputeTarget
from azureml.exceptions import ComputeTargetException
from ml_service.utils.environment_variables import ENV

def get_compute(workspace,
                compute_name,
                vm_size,
                for_batch_scoring=False):
    try:
        if compute_name in workspace.compute_targets:
            compute_target = workspace.compute_targets[compute_name]
            if compute_target and type(compute_target) == AmlCompute:
                print(f'Found existing compute_target: {compute_name}!')
        else:
            e = ENV()
            compute_config = AmlCompute.provisioning_configuration(vm_size=vm_size,
                                                                   vm_priority=e.vm_priority if not for_batch_scoring else e.vm_priority_scoring,
                                                                   min_nodes = e.min_nodes if not for_batch_scoring else e.min_nodes_scoring,
                                                                   max_nodes = e.max_nodes if not for_batch_scoring else e.max_nodes_scoring,
                                                                   idle_seconds_before_scaledown='300')
            compute_target = ComputeTarget.create(workspace=workspace,
                                                  name=compute_name,
                                                  provisioning_configuration=compute_config)
            compute_target.wait_for_completion(show_output=True,
                                               timeout_in_minutes=10)
        return compute_target
    except ComputeTargetException as error:
        print(f'an error occurred {error}')
        exit(1)
