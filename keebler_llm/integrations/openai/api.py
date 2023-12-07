import  pandas as pd 
import  os 
from    typing import List, Dict, Generator, Any 

import  openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,        # for exponential backoff
)  

from    ...core.utils import read_env 

def read_credentials(user_input:bool=False) -> str:
    if user_input:
        openai.api_key = getpass.getpass("OpenAI API Key: ")
    else:
        read_env('.env')
        openai.api_key = os.getenv('OPENAI_API_KEY')

    return openai.api_key


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def generate_chat_completion(model_name:str, messages:dict, config:dict, stream:bool=True):
    """Generate Chat completion task

    Args:
        model_name (str): model name
        messages (dict): messages to instruct
        config (dict): configuration for generation
        stream (bool, optional): streaming data. Defaults to True.

    Returns:
        _type_: _description_

    Examples:
    >>> prompt   = ... 
    >>> messages = [{"role":"user", "content":prompt}]
 
    # response.choices[0].message.content
    # pd.DataFrame( [dict(response.choices[0].message)|dict(response.usage)]) 
    """
    response = openai.ChatCompletion.create(
        model    = model_name,
        messages = messages, 
        stream   = stream, 
        **config
    )
    return response
