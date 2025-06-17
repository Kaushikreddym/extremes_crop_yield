from omegaconf import OmegaConf

cfg = OmegaConf.load("conf/config_datasets.yaml")

all_names = []
for category in ['tas']:
    for index in cfg.indices[category]:
        all_names.append(index.name)

for name in sorted(set(all_names)):
    print(name)