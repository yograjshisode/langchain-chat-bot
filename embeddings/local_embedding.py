# SentenceTransformerEmbeddings can compute sentence / text embeddings for more than 100 language
# This framework generates embeddings for each input sentence Sentences are passed as a list of string.
# We can perform embeddings using OpenAI as well
# Supportted embedding models are listed here: https://www.sbert.net/docs/pretrained_models.html
# https://huggingface.co/spaces/mteb/leaderboard
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Vectore Stores
# Facebook AI Similarity Search is a vectore store
# there are some alternatives like Choroma
# All the langchain supportted vectorestores/dbs are available here
#  https://github.com/langchain-ai/langchain/tree/25ba7332185e0c6624a2b02b72030f073755d716/libs/community/langchain_community/vectorstores
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
# Loaders
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from loaders.loader import Loader

class LocalEmbedding:
    """LocalEmbedding to embed and store documents locally
    To use, you should have the ``sentence_transformers`` python package installed.
    
    Example:
        .. code-block:: python

            from local_embedding import LocalEmbedding

            source_files_path = "../guides"
            local_embedding_path = "../embedding_index"

            le = LocalEmbedding(source_files_path, local_embedding_path)
            le.load_embeddings("Directory")
    """
    def __init__(self, source_files_path: str, local_embedding_path: str, vstore_type: str, model_name="all-mpnet-base-v2"):
        """Initialization
        Args:
            source_files_path (str): Path to the source documents
            local_embedding_path (str): Path to load/store local embeddings
            vstore_type (str): Supportted vstore types [Chroma, FAISS]
            model_name (str, optional): _description_. Defaults to "all-mpnet-base-v2".
        """
        self.source_files_path = source_files_path
        self.local_embedding_path = local_embedding_path
        self.model_name = model_name
        self.vstore = self._get_vectorestore()
        self.vectorestore = None

    def _get_vectorestore(self, vstore_type: str):
        """_summary_

        Args:
            vstore_type (str): Supportted vstore types [Chroma, FAISS] 

        Raises:
            Exception: Raise exception if vstore_type is not suppotted

        Returns:
            VectorStore: VectorStore
        """
        if vstore_type == "Chroma":
            return Chroma
        elif vstore_type == "FAISS":
            return FAISS
        else:
            raise Exception(f"Vectore Store {vstore_type} not supportted")

    def load_embeddings_from_disk(self, embedding: SentenceTransformerEmbeddings):
        """Loads prebviously embedded directly from disk
        Args:
            embedding (SentenceTransformerEmbeddings): Haggingface Embedding Model
        """
        print(f"Loading local FAISS index from {self.local_embedding_path}")
        self.vectorestore = FAISS.load_local(self.local_embedding_path, embedding,
                                                 allow_dangerous_deserialization=True)
        print("done.")

    def load_and_embed_docs(self, loader_type: str, embedding: SentenceTransformerEmbeddings):
        """Load the documents from source_files_path and store embedding in vectore stores
        Args:
            loader_type (str): Any langchain supportted loader
            embedding (SentenceTransformerEmbeddings): Haggingface Embedding Model
        """
        print("Loading documents")
        loader = Loader(self.source_files_path, loader_type)
        loader.load_source()
        print(f"Building FAISS index from documents")
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=75
                )
        frags = text_splitter.split_documents(loader.docs)
        print(f"Poplulating vector store with {len(loader.docs)} docs in {len(frags)} fragments")
        self.vectorestore = FAISS.from_documents(frags, embedding)
        print(f"Persisting vector store to: {self.local_embedding_path}")
        # Save embedded docs locally
        self.vectorestore.save_local(self.local_embedding_path)
        print(f"Saved FAISS index to {self.local_embedding_path}")

    def load_embeddings(self, loader_type: str):
        """Load the embeddings if already exists at local_embedding_path
        otherwise load the documents from source_files_path and stores
        the embeddings in vecorestore
        Args:
            loader_type (str): Supportted loader types [Directory, PDF]
        """
        # We can use force_download=True to force a new download for model
        embedding = SentenceTransformerEmbeddings(model_name=self.model_name)
        # print(f"Embedding: {embedding.e}")
        if os.path.exists(self.local_embedding_path):
            self.load_embeddings_from_disk(embedding)
        else:
            self.load_and_embed_docs(loader_type, embedding)