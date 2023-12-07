import  numpy as np 
import  pandas as pd 
import  enum 
from    typing import List, Union, Optional 
from    pathlib import Path 
from    omegaconf import DictConfig

import  langchain
import  langchain.document_loaders as ldl 
import  langchain.text_splitter as lts
from    langchain.schema import Document
from    ...core.io.utils import is_url_remote, is_valid_url, search_files, search_files_to_dataframe, get_uri_properties

class Loader(object):
    @classmethod
    def load_content(cls, uri:str, root_dir:str, **kwargs) -> List[Document]:
        properties:dict = get_uri_properties(uri, root_dir)
        mapping_svc:str = ('directory' if properties.is_dir else 'extension') if not properties.is_remote  else 'remote'
        mapping:dict    = Loader.lookup(mapping_svc, properties)
        fn_loader       = mapping.get('cls_loader')
        kwargs.update(mapping.get('params', dict()))
        loader          = fn_loader(properties.uri, **kwargs)
        documents       = loader.load()
        return documents

    @classmethod
    def lookup(cls, svc_type:str, properties:dict) -> dict:
        uri = properties.uri 
        ext_lookup = dict(
            text        = dict(cls_loader=ldl.TextLoader), 
            py          = dict(cls_loader=ldl.PythonLoader), 
            csv         = dict(cls_loader=ldl.CSVLoader),
            pdf         = dict(cls_loader=ldl.OnlinePDFLoader if (is_url_remote(uri) and is_valid_url(uri)) else ldl.PyPDFLoader)
        )
        svc_lookup = dict(
            notion_dir  = dict(cls_loader=ldl.NotionDirectoryLoader),
            notion_db   = dict(cls_loader=ldl.NotionDBLoader) 
        )
        defaults_lookup = dict(
            remote      = dict(cls_loader=ldl.WebBaseLoader),  
            directory   = dict(
                cls_loader=ldl.DirectoryLoader,
                params=dict(
                    glob =  f"**/*{properties.ext_suffix}", 
                    recursive=True, 
                    loader_cls=ldl.TextLoader, 
                    use_multithreading=True, 
                    show_progress=False
                )
            ), 
        )
        lookup_svc = ext_lookup | svc_lookup | defaults_lookup
        return ext_fn_loader.get(properties.ext_name) if svc_type == 'extension' else lookup_svc.get(svc_type)



class TransformerChunks(object):
    @classmethod
    def trsfrm_to_text(cls, pages:List[Document]) -> str:
        """Combines the page content from a list of documents into a single string.

        Args:
            pages (list): A list of Document objects representing pages.

        Returns:
            str: A string containing the combined page content.
        """
        all_text = [p.page_content for p in pages]
        return " ".join(all_text)

    @classmethod
    def trsfrm_to_frame(cls, documents:List[Document]) -> pd.DataFrame:
        """transform a series of documents of metadata and page content into a pandas dataframe

        Args:
            documents (List[Document]): loaded documents

        Returns:
            pd.DataFrame: pandas dataframe with metadata and page content
        """
        return pd.DataFrame(
            [dict(metadata=doc.metadata, content=doc.page_content) 
            for doc in documents]
        )

    @classmethod
    def trsfrm_to_chunks(cls, content:Union[List[Document], str], splitter_type:str='recursive', **kwargs) -> List[Document]: 
        """chunk documents rather than the entire document

        Ideally we only want to extract the most relavant sections of the document, not entire document

        Args:
            content (Union[List[Document], str]): List of Documents from Extraction or a text string
            splitter_type (str, optional): mapping of splitter type. Defaults to 'character'.

        Returns:
            List[Document]: List of splitted documents
        """
        chunk_size    = kwargs.get("chunk_size", 1024)
        chunk_overlap = kwargs.get("chunk_overlap", 0)
        params        = dict(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # perform lookup and execution of the splitter 
        mapping:dict  = TransformerChunks.lookup(splitter_type)
        params.update(mapping.get('params', dict()))
        splitter = mapping.get('cls_loader')(**params)
        return splitter.split_text(content) if isinstance(content[0], str) else splitter.split_documents(content)


    @classmethod 
    def lookup(cls, splitter_type:str='recursive') -> dict:
        chunk_loader    = dict(
            md          = dict(cls_loader=lts.MarkdownTextSplitter),
            token       = dict(cls_loader=lts.TokenTextSplitter),
            character   = dict(cls_loader=lts.CharacterTextSplitter, params=dict(separator=' ', length_function=len)),
            recursive   = dict(cls_loader=lts.RecursiveCharacterTextSplitter, params=dict(seperators=["\n\n", "\n", "(?<=\. )", " ", ""])),
        )
        chunk_loader.update(dict(
            md_header   = dict(
                cls_loader=lts.MarkdownHeaderTextSplitter, 
                input_type=str,
                params=dict(
                    headers_to_split_on = [
                    ("#", "Header 1"),
                    ("##", "Header 2"),
                    ("###", "Header 3"),
                ])
            )
        ))
        return chunk_loader.get(splitter_type)


