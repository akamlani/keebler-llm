import  pandas as pd 
import  os
import  getpass
from    typing import List, Dict, Generator, Any 

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,        # for exponential backoff
)  

from    ...core.utils import read_env 
from    ...core.services.rest import service_request



def read_credentials(user_input:bool=False) -> str:
    if user_input:
        openai.api_key = getpass.getpass("OpenAI API Key: ")
    else:
        read_env('.env')
        openai.api_key = os.getenv('OPENAI_API_KEY')

    return openai.api_key


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
def generate_chat_completion_rest(model_name:str, messages:dict, config:dict, endpoint:str, credentials:dict):
    """envoke completion API via RESTful request

    Examples:
    >>> config = {
            "temperature":      temperature,
            "max_tokens":       max_tokens
        }
    # model="gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"
    >>> response_data = service_request(model_name="gpt-3.5-turbo", endpoint=endpoint, body=payload, headers=headers)    
    >>> response_data["choices"][0]["message"]["content"]
    """
    headers={
        "Accept":           "application/json, text/plain",
        "Authorization":    f"Bearer {credentials['api_key']}",
    }

    config["model"]    = model_name
    config["messages"] = messages
    endpoint_default = "https://api.openai.com/v1/chat/completions"
    return service_request(
        endpoint=endpoint_default if not endpoint else endpoint, 
        body=config, 
        headers=headers, 
        session_en=False,
    )
