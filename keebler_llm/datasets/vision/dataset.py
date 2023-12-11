import  numpy as np 
import  pandas as pd 
from    typing import Tuple, List, Optional 
from    pathlib import Path 
from    PIL import Image
from    ...core.io.utils import is_url_remote, is_valid_url


class DatasetImage(object):
    """Loads attributes of image files, loads and normalizes them for processing

    Args:
        object (_type_): _description_

    Examples:
    >>> img_url     = "https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    >>> dataset_dev = data_path.joinpath('train', 'images')
    >>> dataset     = DatasetImage(dataset_dev, target_dim=(224,224,3), resize=False)

    >>> ex_img = dataset.load_image(img_url, resize=False, dimensions=(224,224, 3))
    >>> ex_img = dataset.normalize(ex_img)
    >>> plot_image(ex_img.squeeze(), plt.gca(), **{'title': f"Spatial dimensions: {ex_img.squeeze().shape}"})
    """
    def __init__(self, uri:str, target_dim:Tuple[int,int,int]=None, resize:bool=False):
        self.target_dim  = target_dim
        # ONNX expects NCHW input, so convert the array
        self.ch_formats  = ['NCHW', 'NHWC']  
        self.resize      = resize 
        self.data        = (
            self.get_dataset_attr(uri)
            .apply(self.transform_record, axis=1)
        )

    def transform_record(self, ds:pd.Series):
        orig_img           = self.load_image(ds['uri'], self.target_dim, resize=self.resize)
        ds['dim:(h,w,ch)'] = orig_img.squeeze().shape
        return ds 

    def get_dataset_attr(self, uri:str, pattern:str='*') -> pd.DataFrame: 
        "walk directories to extract files and attributes of files"
        files  = Path(uri).rglob(pattern)
        return pd.DataFrame(files, columns=['uri'])

    def load_image(self, uri:str, dimensions:Tuple[int,int,int]=None, resize:bool=False, channel_format="NHWC"):
        is_remote = lambda s: is_url_remote(s) and is_valid_url(s)

        uri  = requests.get(uri, stream=True).raw if is_remote(uri) else uri 
        img  = Image.open(uri)
        img  = img.resize(dimensions) if resize else img
        img  = np.asarray(img).astype("float32")
        img  = np.transpose(img, (2, 0, 1)) if channel_format == 'NCHW' else img 
        img  = np.expand_dims(img, axis=0)
        return img

    def normalize(self, image:np.array) -> np.array:
        return image / 255.

