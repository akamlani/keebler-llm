import  numpy  as np 
import  pandas as pd 
import  torch 
from    typing import List
from    sentence_transformers import SentenceTransformer, CrossEncoder, util

class DenseEncoder(object):
    # 'paraphrase-MiniLM-L6-v2'
    # https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
    def __init__(self, model_name:str='multi-qa-MiniLM-L6-cos-v1', max_seq_len:int=256, device:str='cpu'):
        self.device_gpu:bool = torch.cuda.is_available()
        self.max_seq_len:int = max_seq_len                                 # truncation
        self.encoder         = SentenceTransformer(model_name).to(device)  # bi-encoder (dim: 384)
        self.encoder.max_seq_length = self.max_seq_len

    def fit(self, texts:List[str], show_progress:bool=True) -> torch.Tensor:
        self.data:List[str] = texts
        return self.encoder.encode(texts, convert_to_tensor=True, show_progress_bar=show_progress)

    def get_lengths(self, texts:List[str]):
        return [self.encode(text)['input_ids'].shape[1] for text in texts]

    def normalize_l2(self, embeddings:np.ndarray) -> np.ndarray:
        # L2 Normalize the rows
        return embeddings / np.sqrt((embeddings**2).sum(1, keepdims=True))

    def encode(self, text:str, convert_to_tensor:bool=True, show_progress:bool=True) -> torch.Tensor:
        emb = self.encoder.encode(text, convert_to_tensor=convert_to_tensor, show_progress_bar=show_progress)
        return emb if not self.device_gpu else emb.cuda()

    def score(self, query_emb:torch.Tensor, corpus_emb:torch.Tensor, top_k:int=100) -> List[List[dict]]:
        # top_k: number of records to retrieve
        hit_at_k =  util.semantic_search(query_emb, corpus_emb, top_k=top_k)
        return hit_at_k

    def lookup(self, df_src:pd.DataFrame, scores:List[dict]) -> pd.DataFrame:
        # for a particular individual query
        return (
            df_src
            .join(pd.DataFrame(scores).set_index('corpus_id'))
            .sort_values(by='score', ascending=False)
            .rename(columns={'score':'dense_score'})
        )
