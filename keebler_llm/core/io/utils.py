import  pandas as pd 
import  re

from    pathlib import Path
from    typing import List, Optional 
from    urllib.parse import urlparse
from    omegaconf import DictConfig

import  logging

logger = logging.getLogger(__name__)


def search_files(uri:str, extension:str) -> List[str]:
    """recursively searches for files matching the extension

    Args:
        uri (str): path to directory
        extension (str): extension pattern to search for

    Returns:
        List[str]: list or string uri's
    """
    return  sorted( map(str, Path(uri).rglob('*' + extension) ) )

def search_files_to_dataframe(uri:str, extension:str) -> pd.DataFrame:
    """generates a dataframe of recurivesly found files matching extension

    Args:
        uri (str): path to directory
        extension (str): extension pattern to search for

    Returns:
        pd.DataFrame: dataframe with full uri and separate filename attributes
    """
    df = (pd.DataFrame(search_files(uri, extension), columns=['uri'])
        .assign(filename=lambda df: df.uri.apply(lambda uri: Path(uri).name) ) \
        .sort_values(by=['filename'], ascending=True).reset_index(drop=True)
    )
    return df 

def list_matching_files(parent_dir: str, pattern: str) -> List[str]:
    """list matching files in a directory from a pattern

    Args:
        parent_dir (str): parent directory to seed from
        pattern (str): pattern to match

    Returns:
        List[str]: list of files that matched the pattern

    Examples
    >>> list_matching_files(exp_data_export_dir, os.path.join("train", "*.parquet"))
    """
    return sorted(parent_dir.glob(pattern))

def get_uri_properties(uri:str, root_dir:Optional[str]=None) -> DictConfig:
    """get properties of a uri, either as a local or remote resource

    Args:
        uri (str): path to directory
        root_dir (Optional[str], optional): root directory for local files. Defaults to None.

    Returns:
        DictConfig: Hydra DictConfig of properties
    """
    is_remote  = is_url_remote(uri) and is_valid_url(uri)
    if is_remote: 
        properties = dict(is_remote=is_remote, uri=uri)
    else: 
        uri        = f"{root_dir}/{uri}" if not Path(uri).exists() else uri
        is_dir     = Path(uri).is_dir()
        ext_suffix = Path(uri).suffix if not is_dir else Path(uri).glob("**/*").__next__().suffix 
        properties = dict(
            uri        = uri,
            is_remote  = is_remote,
            is_file    = Path(uri).is_file(), 
            is_dir     = is_dir,
            ext_suffix = ext_suffix, 
            ext_name   = ext_suffix.lstrip('.')
        )
    return DictConfig(properties)     

def trsfrm_col_camelcase_to_snakecase(col: str) -> str:
    """Transforms column naming from camelcase to snakecase

    Args:
        col (str): input column name to transfrom

    Returns:
        str: transformed column
    """
    column = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', col)
    column = re.sub('([a-z0-9])([A-Z])',  r'\1_\2', col).lower()
    return column.replace(" ", "_")

def trsfrm_frame_camelcase_to_snakecase(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms column naming from camelcase to snakecase for a dataframe

    Args:
        df (pd.DataFrame): input dataframe with columns

    Returns:
        pd.DataFrame: transformed pandas dataframe
    """
    df.columns = map(trsfrm_col_camelcase_to_snakecase, df.columns)
    return df


def trsfrm_normalize_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """renames columns given a dictionary mapping 

    Args:
        df (pd.DataFrame): input frame
        mapping (dict): maaping from dictionary in src:dest format

    Returns:
        pd.DataFrame: _description_
    """
    return df.rename(columns=mapping) if mapping else df

def is_valid_url(uri: str) -> bool:
    """Validates if url is valid

    Args:
        uri (str): url

    Returns:
        bool: validation performance
    """
    try:
        result = urlparse(uri)
        # check for a limited version of supported formats
        return result.scheme in ['http', 'https'] and all([result.scheme, result.netloc])
    except ValueError:
        logger.exception(f"Exception Occured Reading url: {uri}:{e}")
        return False


def is_url_remote(uri: str) -> bool:
    """Check if the uri is local filesystem or remote 

    Args:
        uri (str): uri to parse

    Returns:
        bool: returns True if remote url
    """
    try:
        result = urlparse(uri)
        return (True if result.scheme and result.scheme != "file" else False)
    except ValueError:
        logger.exception(f"Exception Occured Reading url: {uri}:{e}")
        return False
