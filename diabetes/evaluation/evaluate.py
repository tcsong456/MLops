from azureml.core import Run
from utils.model_utils import get_model
import argparse
import traceback

parser = argparse.ArgumentParser('evaluate')
arg = parser.add_argument
arg('--model-name',type=str,default='diabetes_model.pkl',
    help='the name of the model')
arg('--run-id',type=str,
    help='running id of training')
arg('--allow-run-cancel',type=str,default='true',
    help='set to true to cancel unsuccessful evalutation run')
args = parser.parse_args()

run = Run.get_context()
exp = run.experiment
workspace = exp.workspace
run_id = 'amlcompute'

if args.run_id is not None:
    run_id = args.run_id
if args.run_id == 'amlcompute':
    run_id = run.parent.run_id
model_name = args.model_name
metric_eval = 'mse'
allow_run_cancel = args.allow_run_cancel

try:
    model = get_model(model_name=model_name,
                      workspace=workspace,
                      tag_name='experiment_name',
                      tag_value=exp.name)
    if model is not None:
        model_mse = None
        if metric_eval in model.tags:
            model_mse = float(model.tags[metric_eval])
        new_run_mse = float(run.parent.get_metrics().get(metric_eval))
        if model_mse is None or new_run_mse is None:
            if allow_run_cancel == 'true':
                run.parent.cancel()
        else:
            print(f'model mse: {model_mse}\nnew run mse: {new_run_mse}')
        if new_run_mse < model_mse:
            print('current run has better result than previous one,therfore you can continue')
        else:
            print('current run has worse result than previous one,evaluation should hault!')
            if allow_run_cancel == 'true':
                run.parent.cancel()
    else:
        print('no model is registered yet!')
except Exception:
    traceback.print_exc(limit=None,file=None,chain=True)
    raise
