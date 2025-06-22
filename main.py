import hydra
from omegaconf import DictConfig
import warnings
import logging


@hydra.main(config_path="./config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    启动函数
    """
    logging.info("Application Launch")

    warnings.warn("The startup program has not been implemented yet")


if __name__ == "__main__":
    main()
