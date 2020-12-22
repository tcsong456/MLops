from azureml.core.webservice import AksWebservice,AciWebservice
from azureml.core import Workspace
from ml_service.utils.environment_variables import ENV
import secrets
import requests
import time
import argparse

input = {"data": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]}
output_len = 2

def call_web_service(e,service_type,service_name):
    aml_workspace = Workspace.get(name=e.workspace_name,
                                  subscription_id=e.subscription_id,
                                  resource_group=e.resource_group)
    print('fetching webservice')
    if service_type == 'AKS':
        service = AksWebservice(aml_workspace,service_name)
    elif service_type == 'ACI':
        service = AciWebservice(aml_workspace,service_name)
    else:
        raise ValueError(f'no {service_type} is supported!')
    
    headers = {}
    if service.auth_enabled:
        service_keys = service.get_keys()
        headers['Authorization'] = 'Bearer ' + service_keys[0]
    
    scoring_url = service.scoring_url
    print(f'scoring url: {scoring_url}')
    output = call_web_app(scoring_url,headers)
    
    return output

def call_web_app(url,headers):
    headers['traceparent'] = f'00-{secrets.token_hex(16)}-{secrets.token_hex(8)}-00'
    retries = 600
    for i in range(retries):
        try:
            response = requests.post(url,json=input,headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if i == retries - 1:
                raise Exception(e)
            print(e)
            print('Retrying...')
            time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--service-type',type=str,
        choices=['ACI','AKS','Webapp'],
        required=True,
        help='specify the service type')
    arg('--service-name',type=str,
        required=True,
        help='specify the name of the service')
    args = parser.parse_args()
    
    e = ENV()
    service_type = args.service_type
    if service_type == 'Webapp':
        output = call_web_app(args.service_name,{})
    else:
        output = call_web_service(e,service_type,args.service_name)
    print('Verifying output validlity')
    assert 'result' in output
    assert len(output) == output_len
    print(output)
    print('Smoke test successful!')
    
    

