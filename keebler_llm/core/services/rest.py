import requests 
import json 
import logging 
import time 
from   typing import Optional, Dict 
from   collections import defaultdict
from   dataclasses import dataclass

logger = logging.getLogger(__name__)

def service_request(
    endpoint:str, 
    body:dict, 
    headers:dict=None,
    session_en:bool=False, 
    **kwargs
) -> Optional[Dict[str, str]]:
    """Send Post Request to Server

    Args:
        endpoint (str): endpoint url
        body (dict): payload for endpoint
        session_en (bool, optional): if session should be enabled. Defaults to False.
        user_agent (str, optional): calling applicaation. Defaults to 'AppClient/1.0'.

    Returns:
        Optional[Dict[str, str]]: response from the endpoint, transformed in json format

    Examples:
    >>> header={
            "User-Agent":       "RuntimeApp/1.0",
            "Accept":           "application/json, text/plain",
            "Connection":       "keep-alive", 
        }
    >>> response_data = service_request(endpoint=endpoint, body=payload, headers=headers)
    """
    try:
        headers = dict(defaultdict(str)) if not headers else headers
        header_defaults = {
            "Accept":        "application/json",
            "Cache-Control": "no-cache",
            "User-Agent":    "ServiceRuntime/1.0"
        }
        
        headers = {**header_defaults, **headers}
        if body is not None: 
            headers.update({
                "Content-Type":  "application/json", 
                'Content-Length': str(len(json.dumps(body))),
            })

        context_fn = requests.Session().post if session_en else requests.post
        # not required to use json.dumps(body), since we are using kw 'json='
        response   = context_fn(endpoint, json=body, headers=headers)
        response.raise_for_status()
        # check for 200 code
        if response.status_code == requests.codes.OK:
            return response.json() 
        else: 
            logger.error(f"Error Occured with Endpoint: {endpoint}:{response.status_code}")
            logger.error(f"Content from Response: {response.text}")

    except requests.exceptions.RequestException as e:
        logger.exception(f"Exception Occured with Endpoint: {endpoint}:{e}")
        return None 





