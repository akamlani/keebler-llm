import multiprocessing
import psutil 
import functools
import time 
import logging
from   typing import Callable, Any, List

logger = logging.getLogger(__name__)


def profile_device(): 
    return dict(
        num_cores    = multiprocessing.cpu_count()
    )


def profile_memory(gb_unit:bool=True) -> dict:
    # convert to either GB or MB 
    scale  =  (1024.0 ** 3) if gb_unit else  (1024.0 ** 2)
    memory = psutil.virtual_memory()
    return dict(
        memory_units     = 'GB' if gb_unit else 'MB',
        total_memory     = round(memory.total / scale, 3),  
        available_memory = round(memory.available / scale, 3), 
        used_memory      = round(memory.used / scale, 3)
    )

def get_data_sz_mb(data:List[str]):
    return sum(len(s.encode("utf-8")) for s in data) / 1024 / 1024
