import  numpy as np 
import  pandas as pd 
from    typing import Any, Optional 

from    typing import TypeVar, Generic
T_co    = TypeVar('T_co', covariant=True)
T       = TypeVar('T')

class DatasetT(Generic[T_co]):
    def __init__(self, **kwargs):
        pass 

    def __getitem__(self, index:int):
        sample = None 
        if self.transform:
            sample = self.transform(sample)

        return sample[index]
        
    def __len__(self):
        try: 
            len(self.data)
        except e: 
            raise NotImplementedError
    
    def info(self, df:pd.DataFrame) -> dict:
        raise NotImplementedError

    def load(self, path:str) -> None:
        raise NotImplemented

    def transform(self, data:Any, target:Any=None, **kwargs) -> None:
        raise NotImplementedError

    def validate(self, schema:dict, **kwargs) -> bool:
        raise NotImplementedError

    def transform(self, data:np.array, **kwargs) -> Any:
        raise NotImplementedError


class DatasetTabular(DatasetT):
    def __init__(self, df:pd.DatFrame, target:Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.data    = data
        self.indices = df.index 
        self.num_obs, self.num_features = df.shape
        if target:
            self.target = target 
            self.target_cardinality = len( np.unique(target) )

    def __repr__(self):
        return f"Class: {self.__class__.__name__} | Shape: {self.data.shape}"


