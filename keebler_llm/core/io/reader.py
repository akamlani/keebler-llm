import  pandas as pd 
import  hydra 
import  yaml 
import  logging
import  functools

from    pathlib import Path
from    typing import TypeVar, Callable, Optional, List, Dict, Tuple, Any
from    omegaconf import DictConfig, OmegaConf

from    .utils import is_valid_url

logger = logging.getLogger(__name__)


# define potential return types for the decorating function
T = TypeVar('T', pd.DataFrame, DictConfig,
            Dict[str, Any], List[Any], Any, None)


def read_exec_io(func: Callable[[str, Tuple[Any, ...], Dict[str, Any]], T]
                 ) -> Callable[[str, Tuple[Any, ...], Dict[str, Any]], Optional[T]]:
    @functools.wraps(func)
    def wrapper(path: str, *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Optional[T]:
        try:
            if Path(path).is_file() or is_valid_url(path):
                return func(path, *args, **kwargs)
            else:
                logger.error(f"Path: {path} is malformed or does not Exist")
                return None
        except Exception as e:
            logger.exception(f"Exception Occured Reading File: {path}:{e}")
            return None
    return wrapper


@read_exec_io
def read_yaml(filepath: str, encoding: str = "utf-8",  *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Optional[dict]:
    """Reads a Yaml file for usage primarily with configuration

    Args:
        filepath (str): path to location of file to read
        encoding (str, optional): encoding representation of file. Defaults to "utf-8".

    Returns:
        Optional[dict]: dictionary of contents read
    """
    with open(filepath, encoding=encoding) as f:
        data: dict = yaml.load(f, Loader=yaml.FullLoader)
        return data

@read_exec_io
def read_hydra(filepath: str,  *args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Optional[DictConfig]:
    """Reads a Hyrda Yaml Configuraiton

    Args:
        filepath (str): path to location of file to read

    Returns:
        Optional[DictConfig]: Structured Dictionary of OmegaConf
    """
    return OmegaConf.load(filepath)

@read_exec_io
def read_json(filepath: str, encoding: str = "utf-8",  *args: Tuple[Any, ...],
              **kwargs: Dict[str, Any]) -> Optional[dict]:
    """reads a JSON file 

    Args:
        filepath (str): path to location of file to read
        encoding (str, optional): encoding representation of file, default to utf-8. Defaults to "utf-8".

    Returns:
        Optional[dict]: dictionary of contents read from file 
    """
    with open(filepath, encoding=encoding) as f:
        data: dict = json.load(f)
        return data


@read_exec_io
def read_jsonl_to_pandas(filepath: str,  *args: Tuple[Any, ...], **kwargs: Dict[str, Any]
                              ) -> Optional[pd.DataFrame]:
    """Reads lines encoded in JSON format (*.jsonl) into Pandas format

    Args:
        filepath (str): path to location of file to read

    Returns:
        Optional[pd.DataFrame]: pandas dataframe of jsone lines format read

    Examples:
    >>> df_dataset = .read_jsonl_to_pandas(filepath="docs.jsonl")
    >>> examples   = df_dataset.to_dict()
    >>> df_dataset.shape
    """
    return pd.read_json(filepath, orient='records', lines=True)

