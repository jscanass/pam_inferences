import argparse
from yaml import safe_load
from pipelines.WorkFlow import Workflow

parser = argparse.ArgumentParser(description='Inferences')
parser.add_argument('--config', help='Path to config file', default='configs/example.yaml')
args = parser.parse_args()

print(f'Using config "{args.config}"')

cfg = safe_load(open(args.config, 'r'))

site = cfg['site']
model_parameters = cfg['model']
trained_path = model_parameters['trained_path']
hyper_parameters = model_parameters['hyper_parameters']

wf = Workflow(site=site, model=trained_path, metadata=hyper_parameters)
wf.run()