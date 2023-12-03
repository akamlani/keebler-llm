import numpy as np 
import pandas as pd
import scipy.stats as scs 

def calc_zscore(data:np.array) -> np.array:
    # does not perform well on small datasets, based on extreme values
    return (data - np.mean(data))/np.std(data)

def calc_zscore_is_outlier(data:np.array, threshold:float=3.0) -> np.array:
    z = calc_zscore(data)
    return np.where(np.abs(z) > threshold, True, False).astype(int)

def calc_iqr(data:np.array) -> float:
    q25, q75 = np.percentile(data, [25 ,75])
    iqr      = q75-q25
    return iqr 

def calc_iqr_is_outlier(data:np.array) -> np.array:
    q25, q75     = np.percentile(data, [25 ,75])
    iqr          = q75-q25    
    lower, upper = ( (q25 - (1.5*iqr)), (q75 + (1.5*iqr)) )
    outliers = (data < lower) | (data > upper)
    return outliers.astype(int) 

def calc_zscore_ci(data:np.array, confidence_interval:float=0.95) -> (float, float):
    mean = data.mean()
    std  = data.std()

    z_lower = scs.norm.ppf((1 - confidence_interval) / 2)
    z_upper = scs.norm.ppf(1 - (1 - confidence_interval) / 2)
    threshold_lower = mean + z_lower * std
    threshold_upper = mean + z_upper * std
    return (threshold_lower, threshold_upper)

def calc_outliers(df:pd.DataFrame, col:str, threshold:float=3):
    # threshold: [3,2.5,2,1.5,1]
    return df[col].to_frame().assign(
        zscore         = lambda df_: calc_zscore(df_[col]),
        zscore_scs     = lambda df_: scs.zscore(df_[col]), 
        zscore_outlier = lambda df_: calc_zscore_is_outlier(df_[col], threshold=threshold), 
        iqr_outlier    = lambda df_: calc_iqr_is_outlier(df_[col]),
    )

