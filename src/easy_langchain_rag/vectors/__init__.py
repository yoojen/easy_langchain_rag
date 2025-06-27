from pathlib import Path
from typing_extensions import Type, List, Union
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import LLMChainFilter, LLMChainExtractor, EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever


class VectorStoreActions:
    def __init__(self,
                 vector_store: Type[FAISS] = FAISS,
                 vector_store_location: str = None,
                 embedding_model: Type[HuggingFaceEmbeddings] = HuggingFaceEmbeddings,
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 save_location: str = None,
                 chunks: List[Document] = None):
        """
        Initialize a VectrorStoreActions object.

        Args:
            vector_store (Type[FAISS]): The vector store to use.
            vector_store_location (str, optional): The location of the existing vector store to load. Defaults to None.
            embedding_model (Type[HuggingFaceEmbeddings], optional): The embedding model to use. Defaults to HuggingFaceEmbeddings.
            embedding_model_name (str, optional): The name of the embedding model to use. Defaults to None.
            save_location (str, optional): The location to save the vector store. Defaults to None.
            chunks (List[Document], optional): The chunks to use when creating the vector store. Defaults to None.
        """
        # Validation checks
        if not vector_store:
            raise ValueError("vector_store must of FAISS type.")
        
        if not embedding_model or embedding_model != HuggingFaceEmbeddings:
            raise ValueError(f"embedding_model must be of HuggingFaceEmbeddings instance. {embedding_model}: {type(embedding_model)}")
        
        if not isinstance(embedding_model_name, str) or not embedding_model_name:
            raise ValueError("embedding_model_name must be a non-empty string.")
        
        if not save_location and not vector_store_location:
            if not save_location or not isinstance(save_location, str):
                raise ValueError("""save_location or vector_store_location is required.If you don't have already processed vector store you need where to save them""")
        
        if save_location and vector_store_location:
            raise ValueError("save_location and vector_store_location cannot be both provided. Provide save_location if you want new embeddings or vector_store_location if you are loading existing vector store")
        
        if save_location:
            path = Path(save_location)
            if not path.exists():
                print("Creating directory: ", save_location)
                path.mkdir(parents=True)
        
        if not vector_store_location and (not save_location or not isinstance(save_location, str)):
            raise ValueError("""save_location or vector_store_location is required.If you don't have already processed vector store you need where to save them""")
        
        if vector_store_location:
            path = Path(vector_store_location)
            if not path.exists():
                raise ValueError("vector_store_location directory does not exist.")
                
        if not vector_store_location and (not chunks or not isinstance(chunks, list)):
            raise ValueError("chunks must be a list of Document objects or None.")
        
        if not chunks or not all(isinstance(doc, Document) for doc in chunks):
            raise ValueError("All items in chunks must be instances of Document.")
        
        self.vector_store = vector_store
        self.vector_store_location = vector_store_location
        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.save_location = save_location
        self.chunks = chunks
        self.embeddings = self.embedding_model(model_name=self.embedding_model_name)

    def _save_vector_store(self):
        """
        Save the vector store to a local directory.
        The directory is created if it does not exist.

        Raises:
            Exception: If any error occurs.
        """
        try:
            path = Path(self.save_location)
            # Create the 'save_location' directory if it does not exists
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

            if path.stat().st_size != 0:
                raise Exception("Directory is not empty")
            
            vector_store = self.vector_store.from_documents(self.chunks, self.embeddings)
            vector_store.save_local(folder_path=self.save_location)
        except Exception:
            raise
    
    def _load_existing_vector_store(self, location=None) -> FAISS:
        """
        Load an existing vector store.

        Args:
            location (str, optional): The location of the vector store to load. Defaults to None.

        Returns:
            The loaded vector store.
        """

        if not location and (not self.vector_store_location and not self.save_location):
            raise Exception("vector_store_location or save_location is required to load vector store.")

        # If location is not specified, use the default location: user specified vector store location
        if not location:
            location = self.vector_store_location or self.save_location # To alow both because might still be using save_location

        try:
            vectorstore = self.vector_store.load_local(folder_path=location, embeddings=self.embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception:
            raise

    def load_vector_store(self):
        """
        Load a vector store from the specified location. If the location is not specified, save the vector store to a local directory
        and then load it. The directory is created if it does not exist.

        Returns:
            The loaded vector store.

        Raises:
            Exception: If any error occurs.
        """

        if self.vector_store_location:
            vectorstore = self._load_existing_vector_store()
        else:
            self._save_vector_store()
            vectorstore = self._load_existing_vector_store(self.save_location)
        if not vectorstore:
            raise Exception("Vector store could not be loaded.")
        
        return vectorstore
    
    def load_vector_store_compressor(
        self,
        llm: Type[Union[BaseChatModel, BaseLLM]],
        retriever: VectorStoreRetriever,
        compressor: Type[Union[LLMChainFilter, LLMChainExtractor,EmbeddingsFilter]] = EmbeddingsFilter,
    ) -> ContextualCompressionRetriever:
        """
        Load the vector store and create a retriever from it. Then, apply a compression technique to the retriever.

        Args:
            llm: The language model to use for the compression technique.
            compressor: The compression technique to use. Defaults to EmbeddingsFilter.

        Returns:
            The compressed retriever.
        """
        # Load vector store
        vector_store = self.load_vector_store()
        if not vector_store:
            raise Exception("Vector store could not be loaded.")

        # Apply compression
        compressor = compressor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

        return compression_retriever
