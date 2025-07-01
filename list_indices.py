from omegaconf import OmegaConf
import hydra
from pathlib import Path

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    all_names = [index.name for index in cfg.mappings.indices]
    
    for name in sorted(set(all_names)):
        print(name)

if __name__ == "__main__":
    main()