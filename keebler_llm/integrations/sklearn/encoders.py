import  numpy  as np 
import  pandas as pd 
from    typing import List

import  sklearn 
from    sklearn.metrics.pairwise import cosine_similarity
from    scipy.sparse._csr import csr_matrix

import  logging 
logger = logging.getLogger(__name__)

class EvaluatorEncoder(object):
    """Evaluate an Encoder, e.g., TFIDF

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    data = {
        'movie': ['Movie A', 'Movie B', 'Movie C', 'Movie D', 'Movie E'],
        'genre': ['Action Adventure', 'Romance Drama', 'Action Sci-Fi', 'Romance Comedy', 'Documentary']
    }
    df = pd.DataFrame(data)

    tfidf         = TfidfVectorizer(stop_words='english')
    tfidf_matrix  = tfidf.fit_transform(df['genre'])
    vocab         = tfidf.get_feature_names_out()

    evaluator = EvaluatorEncoder(encodings=tfidf_matrix, vocab=vocab, indices=df['movie'])
    sparsity  = evaluator.get_sparsity() )
    df_tokens = evaluator.to_frame(df)
    df_scores = evaluator.calc_cosine_sim_frame()
    df_scores['Movie A'].sort_values(ascending=False).to_dict()
    """

    def __init__(self, encodings:csr_matrix, vocab:np.ndarray, indices:np.array):
        self.encodings = encodings
        self.vocab     = vocab
        self.df_vector = pd.DataFrame(self.encodings.toarray(), columns=self.vocab, index=indices)

    def to_frame(self, df:pd.DataFrame) -> pd.DataFrame:
        # e.g, row.nonzero() returns nnz_row_ind, nnz_col_inz
        return df.assign(
            tokens_encoded = [
                sorted([self.vocab[col_idx] for col_idx in row.nonzero()[1]])  
                for row in self.encodings 
            ]
        )

    def calc_cosine_sim_frame(self):
        indices = self.df_vector.index
        cosine_sim_items    = cosine_similarity(self.encodings, self.encodings)
        return pd.DataFrame(cosine_sim_items, index=indices, columns=indices)

    def token_to_index(self, tokens:List[str]) -> np.ndarray:
        return np.where(np.isin(self.vocab, tokens))[-1]

    # for given record, look up token score (tfidf score)
    def lookup_token_score(self, corpus_index:int, tokens:List[str]) -> dict:
        try:
            encodings     = self.encodings.toarray()
            token_indices = list( self.token_to_index(tokens) )
            scores        = encodings[corpus_index, token_indices]
            return {
                token: dict(indices = index, scores = round(score, 3)) 
                for token, index, score in zip(tokens, token_indices, scores)
            }
        except ValueError as e:
            logger.exception(f"Exception, Token not in Vocab: {token}:{e}")
            return None

    def lookup_score(self, encoding:np.ndarray, k:int=5) -> dict:
        # flatten to extract single list: (1, num_rows) -> (num_rows, )
        encodings        = self.encodings.toarray()
        metric_vector    = cosine_similarity(encoding, encodings).flatten()
        sorted_indices   = np.argsort(metric_vector)[::-1].astype(int)
        top_k_indices    = list(sorted_indices[:k])
        results          = [round(metric_vector[idx],3) for idx in top_k_indices] 
        return dict(top_k_ind=top_k_indices, scores=results)

    def get_sparsity(self) -> float:
        return round( (self.encodings.nnz / np.prod(self.encodings.shape)) * 100, 3)

    def calc_similarity_matrix(self) -> np.ndarray:
        """calculates the similarity matrix between a set of rows in a matrix

        Args:
            encoded (csr_matrix): encoded matrix 

        Returns:
            np.ndarray: numpy multidimensional array of shape (n_rows, n_rows)

        Example:
        >>> sim_matrix   = enc.calc_similarity_matrix(encodings)
        metric_vector    = sim_matrix[corpus_index]
        sorted_indices   = np.argsort(metric_vector)[::-1]
        top_k_indices    = sorted_indices[:k]
        results          = [(idx, metric_vector[idx]) for idx in top_k_indices]
        """
        # shape of similarity_matrix will be (n_rows, n_rows)
        return cosine_similarity(self.encodings)
