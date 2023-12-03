import numpy as np 
import pandas as pd 

min_max_scalar_fn = lambda x: (x - min(x)) / (max(x) - min(x))


class DateProperties(object):
    def __init__(self, df:pd.DataFrame, col:str):
        self.min_date = min(df[col])
        self.max_date = max(df[col])

    def filter_dates(self, df:pd.DataFrame, col:str, start_date:str, end_date:str):
        return df[(df[col] >= start_date) & (df[col] <= end_date)]

    # record operation
    def calc_dt_timespan(self, start_time:np.datetime64, end_time:np.datetime64) -> dict:
        # division for years is to account leap years
        calc_dt_duration          = lambda start, end: abs(end - start)
        trsfrm_timedelta_to_years = lambda td: (td / 365.25) 
        trsfrm_timedelta_to_qtrs  = lambda td: (td / (np.timedelta64(1, 'D')) / (30 * 3))

        duration = calc_dt_duration(start_time, end_time)
        return dict(
            days     = round(duration.days, 2),
            quarters = round(trsfrm_timedelta_to_qtrs(duration), 2),
            years    = round(trsfrm_timedelta_to_years(duration.days), 2)
        )    

    # frame operations
    def trsfrm_frame_features_tod(self, df:pd.DataFrame, col:str) -> pd.DataFrame:
        return df.assign(
            day     = lambda df_: df_[col].dt.day,
            month   = lambda df_: df_[col].dt.month,
            quarter = lambda df_: df_[col].dt.quarter, 
            year    = lambda df_: df_[col].dt.year
        )
