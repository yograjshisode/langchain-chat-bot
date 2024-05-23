from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader

class Loader:
    """Loads the documents from different type of sources
        Example:
        .. code-block:: python

            from loader import Loader

            source_files_path = "../guides"
            loader_type = "Directory"

            loaderT = Loader(source_files_path, loader_type)
            loaderT.load_source()
    """
    def __init__(self, source_files_path: str, loader_type: str):
        """

        Args:
            source_files_path (str):  Path to the source documents
            loader_type (str): Path to load/store local embeddings
        """
        self.source_files_path = source_files_path
        self.loader = self._get_loader(loader_type)
        self.docs = None
    
    def _get_loader(self, loader_type: str):
        """Factory method to ge the different loaders

        Args:
            loader_type (str): Path to load/store local embeddings

        Raises:
            Exception: Raise exception if loader_type is not suppotted

        Returns:
            BaseLoader: BaseLoader
        """
        if loader_type == "Directory":
            return DirectoryLoader
        elif loader_type == "PDF":
            return PyPDFLoader
        else:
            raise Exception(f"Loader type {loader_type} not supportted")
    
    def load_source(self, **kwargs):
        """Load document from source_files_path
        """
        loader = self.loader(self.source_files_path, **kwargs)
        self.docs = loader.load()
        return