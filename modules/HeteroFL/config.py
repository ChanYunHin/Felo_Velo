import yaml

global cfg
if 'cfg' not in globals():
    with open('/home/cyh/fed/fed_distill_pytorch/modules/HeteroFL/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
cfg['metric_name'] = {'train': {'Local': ['Local-Loss', 'Local-Accuracy']},
                      'test': {'Local': ['Local-Loss', 'Local-Accuracy'], 'Global': ['Global-Loss', 'Global-Accuracy']}}

        