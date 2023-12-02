import  pandas as pd 
from    typing import List
from    ....core.io.utils import trsfrm_frame_camelcase_to_snakecase

import yfinance as yf 

class DatasetTicker(object):
    def __init__(self, ticker:str, start_date:str, end_date:str):
        self.ticker     = ticker
        self.start_date = start_date 
        self.end_date   = end_date
        self.df         = self.load_from_module(ticker, start_date, end_date)

    def load_from_module(self, ticker:str, start_date:str, end_date:str) -> pd.DataFrame:
        return (
            yf.Ticker(ticker)
            .history(start=start_date, end=end_date)
            .pipe(trsfrm_frame_camelcase_to_snakecase)
        )

    def get_ticker_info(self, ticker:str) -> dict:
        return yf.Ticker(ticker).info()

class DatasetTickers(object):
    def __init__(self, tickers:List[str], start_date:str, end_date:str):
        self.tickers    = tickers
        self.start_date = start_date 
        self.end_date   = end_date
        self.df         = self.load_from_module(tickers, start_date, end_date)

    def load_from_module(self, tickers:list[str], start_date:str, end_date:str):
        return yf.download(tickers, start=start_date, end=end_date, interval='1d') 

    def select_ticker(self, ticker:str) -> pd.DataFrame:
        return (
            self.df.xs(ticker, axis=1, level=1)
                .pipe(trsfrm_frame_camelcase_to_snakecase)
        )