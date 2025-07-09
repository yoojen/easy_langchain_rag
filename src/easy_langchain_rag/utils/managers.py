import os
import hashlib
import logging
from typing_extensions import Type
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

class EmbeddingStoreManager:
    def __init__(self, embedding_path:str, embedding_function: Type[HuggingFaceEmbeddings], allow_dangerous_deserialization=True):
        """
        Initialize a EmbeddingStoreManager object.

        Args:
            embedding_path (str): The path to a FAISS index to load.
            embedding_function (Type[HuggingFaceEmbeddings]): The embedding model to use.
            allow_dangerous_deserialization (bool): Whether to allow deserialization of the index. Defaults to True.
        """
        if not embedding_path:
            raise ValueError("embedding_path is required")
        
        if embedding_path:
            if "../" in embedding_path:
                raise ValueError("embedding_path must be a relative path")
        
            # Associate the filename with the current working directory
            path= Path(os.getcwd())/embedding_path
            if not path.exists():
                raise ValueError(f"file_path {path} does not exist")
                
        if not embedding_function:
            raise ValueError("embedding_function is required")
                
        self.embedding_path = embedding_path
        self.embedding_function = embedding_function
        self.vectorstore = FAISS.load_local(embedding_path, embedding_function, allow_dangerous_deserialization=allow_dangerous_deserialization)
        self.existing_doc_ids = self._load_existing_chunk_ids()
    
    def _load_existing_chunk_ids(self):
        """
        Load a list of existing document chunk ids from the vector_store.

        A document chunk id is a hash of the content of a chunk of a document. This
        function is used to load the existing document chunk ids from the vector_store
        when the EmbeddingStoreManager is initialized.

        Returns:
            list: A list of document chunk ids.
        """
        ids_dict = self.vectorstore.index_to_docstore_id
        return list(ids_dict.values())
    
    def _create_doc_hash(self, chunks):
        """
        Create a list of ids by hashing the content of each chunk in a list of chunks.
        
        Args:
            chunks (list): A list of Document objects.
        
        Returns:
            tuple: A tuple containing a list of ids and the length of the list
        """
        ids = []
        for chunk in chunks:
            chunk_text = " ".join(chunk.page_content.strip())
            has_version = hashlib.sha256(f"{chunk_text}".encode('utf-8')).hexdigest()
            ids.append(has_version)

        return ids
        

    def update_vector_store(self, vector_store: FAISS, chunks: list[Document]):
        """
        Update the given vector store with new or modified document chunks.

        This function compares the existing document chunk ids with newly calculated
        chunk ids from the provided chunks. If a chunk id has not changed, it is skipped.
        New chunks (those whose ids are not present in the existing ids) are added to the
        vector store, and their ids are appended to the list of existing ids. Deleted chunks
        (those whose ids are not present in the new ids) are removed from the vector store,
        and their ids are removed from the list of existing ids.

        Args:
            vector_store (FAISS): The vector store to be updated.
            chunks (list[Document]): A list of Document objects representing the chunks
                                    to be compared and potentially added to the vector store.

        Returns:
            FAISS: The updated vector store.
        """

        new_vector_store = vector_store
        old_doc_ids = self.existing_doc_ids
        new_doc_ids = self._create_doc_hash(chunks)

        for index, (id, new_id) in enumerate(zip(old_doc_ids, new_doc_ids)):
            # If id == new_id, content hasn't changed
            if id == new_id:
                # Just skip it
                logging.info(f"Paragraph {id} has not changed content")
                continue

            # If new_id not in old_doc_ids, it's a new paragraph
            if new_id not in old_doc_ids:
                logging.info(f"Paragraph {new_id} is a new paragraph")
                # add new doc and continue
                new_vector_store.add_documents(documents=[chunks[index]], ids=[new_id])
                # then update the ids list to avoid conflicts
                old_doc_ids.append(new_id)
                continue

            # If id not in new_doc_ids, it's a deleted paragraph
            if id not in new_doc_ids:
                # delete doc and continue
                if new_vector_store.get_by_ids([id]):
                    logging.info(f"Paragraph {id} is a deleted paragraph")
                    new_vector_store.delete(ids=[id])
                
                    # then update the ids list to avoid conflicts
                    old_doc_ids.remove(id)
                else:
                    logging.info(f"Paragraph {id} not found in vector store")
                continue
        
        return new_vector_store
    
    def save_updated_vector_store(self, vector_store: FAISS, index="index"):
        """
        Save the updated vector store to disk.
        
        Args:
            vector_store (FAISS): The vector store to be saved.
        """
        try:
            vector_store.save_local(self.embedding_path, index=index)
            logging.info(f"Vector store saved to {self.embedding_path}")
        except Exception as e:
            logging.error(f"Failed to save vector store: {e}")