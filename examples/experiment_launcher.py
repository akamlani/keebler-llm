import logging
from pathlib import Path

from omegaconf import DictConfig
from rich.logging import RichHandler

import  sys 
sys.path.append('./../')
from keebler_llm.core.io.reader import read_hydra

logger = logging.getLogger(f"PROGRAM:{__name__}")
logger.addHandler(RichHandler(markup=True))

if __name__ == "__main__":
    # load experiment structure to be installed at root
    select_path: str     = Path(__file__).parent.joinpath("..").resolve()
    conf_folder: str     = select_path.joinpath("config").resolve()
    log_conf_path:str    = Path(conf_folder).joinpath("logging.yaml")

    # gather experiment structure
    exp_conf: DictConfig = read_hydra(conf_folder.joinpath("experimentation/experiment.yaml"))
    exp_log_dir: str     = exp_conf.experiment.reporting.logs
    # log information, enable a rich handler instead
    logger.info(f"{__name__} => Logger and Experiment Structure Enabled")


