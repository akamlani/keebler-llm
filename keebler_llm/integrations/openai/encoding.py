import  pandas as pd 
import  tiktoken
from    typing import List, Dict, Generator, Any 

class Tokenizer(object):
    # "text-davinci-003"
    # "gpt-3.5-turbo"
    # "cl100k_base" -> design to work with ada-002 model
    def __init__(self, model_name:str="gpt-3.5-turbo-16k"):
        self.encoding = tiktoken.encoding_for_model(model_name)

    # Tokens are often ~ 4 characters, 2/3 of a word
    def pipe(self, df:pd.DataFrame, col:str='text') -> pd.DataFrame:
        return df.assign(
            words       = lambda df_: df_[col].apply(lambda s: s.split()),
            token       = lambda df_: df_[col].apply(lambda s: self.encoding.encode(s)), 
            word_len    = lambda df_: df_['words'].apply(len),
            token_len   = lambda df_: df_['token'].apply(len),
            decoded     = lambda df_: df_['token'].apply(lambda x: self.encoding.decode(x)),
            token_map   = lambda df_: df_['token'].apply(lambda x: self.decode_encoding(x))
        )
    
    def encode(self, text:List[str]) -> List[int]:
        return self.encoding.encode(text)

    def decode(self, ids:List[int]) -> List[str]:
        return self.encoding.decode(ids)

    def decode_encoding(self, ids:List[int]) -> Dict[int, str]:
        return {token:self.encoding.decode([token]) for token in ids}
