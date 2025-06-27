import os
import logging
from pathlib import Path
from typing_extensions import Type, Union, List
from langchain_core.documents import Document
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import (
    TextLoader, UnstructuredWordDocumentLoader, Docx2txtLoader, PyPDFLoader, PDFPlumberLoader, UnstructuredMarkdownLoader)
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter, MarkdownTextSplitter


class DocumentProcessor:
    def __init__(self,
                 file_path: str,
                 text_splitter: Type[Union[CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter, MarkdownTextSplitter]],
                 document_loader: Type[Union[
                     TextLoader, PyPDFLoader, PDFPlumberLoader, Docx2txtLoader,
                     UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader]
                 ],
                 chunk_size=1000,
                 chunk_overlap=0,
                 separator=" "
                 ):
        """
        Initialize a DocumentProcessor object.

        Args:
            file_path (str): The path to the file to be processed.
            text_splitter (Type[Union[CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter,MarkdownTextSplitter]]):
                The text splitter to use.
            document_loader (Type[Union[TextLoader, PyPDFLoader,PDFPlumberLoader, Docx2txtLoader,
                UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader]]):
                The document loader to use.
            chunk_size (int, optional): The size of each chunk. Defaults to 1000.
            chunk_overlap (int, optional): The overlap between chunks. Defaults to 0.
            separator (str, optional): The separator to use. Defaults to " ".
        """
        if not isinstance(file_path, str):
            raise ValueError("file_path must be a string")
        
        if file_path:
            path= Path(file_path)
            if not path.exists():
                raise ValueError(f"file_path {file_path} does not exist")

        if text_splitter not in [CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter, MarkdownTextSplitter]:
            raise ValueError("Only 'CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter, MarkdownTextSplitter' are supported")
                
        if document_loader not in [TextLoader, PyPDFLoader,PDFPlumberLoader, Docx2txtLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader]:
            raise ValueError("Only 'TextLoader, PyPDFLoader,PDFPlumberLoader, Docx2txtLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader' are supported")
        
        if not isinstance(chunk_size, int) or chunk_size < 100:
            raise ValueError("chunk_size must be an integer and greate than or equal to 100")
        
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            raise ValueError("chunk_overlap must be an positive integer from 0")
        
        if not isinstance(separator, str) or separator not in [" ", "\n", "paragraph"]:
            raise ValueError("separator must be a string")
        
        
        self.knowledge_base = file_path
        self.document_loader = document_loader
        self.text_splitter = text_splitter
        self.chunks = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'md'}

    def _does_file_exists(self):
        """
        Check if the file at the specified path exists.

        Returns:
            True if the file exists, False otherwise.
        """
        file_path = Path(self.knowledge_base)
        return file_path.exists()

    def _validate_document_extension(self):
        """
        Validate that the file extension is valid.
        It checks if the extension is in the list of allowed extensions.

        Returns:
            The filename of the document to be loaded.

        Raises:
            ValueError: If the file extension is not valid.
        """
        filename = secure_filename(self.knowledge_base)
        extension = filename.rsplit(".", 1)[1].lower()

        if extension not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"Invalid file extension: {extension}")

        return filename

    def _load_file(self) -> List[Document]:
        """
        Load the content of the specified file.

        Returns:
            The content of the file as loaded by the text_loader.
        """
        filename = self._validate_document_extension()
        loader = self.document_loader(filename)
        documents = loader.load()

        return documents

    def _split_document(self) -> List:
        """
        Split the loaded document into chunks using the specified text splitter.

        This method uses the text splitter initialized with the specified chunk size,
        overlap, and separator to split the document into manageable chunks.
        The resulting chunks are stored in the instance variable `self.chunks`.

        Returns:
            List: A list of document chunks.
        """
        if self.text_splitter == RecursiveCharacterTextSplitter:
            text_splitter = self.text_splitter(separators=["\n"], chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        elif self.text_splitter == CharacterTextSplitter:
            text_splitter = self.text_splitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator=self.separator)
        else:
            raise ValueError("Unsupported text splitter. Use 'CharacterTextSplitter' or 'RecursiveCharacterTextSplitter'.")
        
        documents = self._load_file()
        chunks = text_splitter.split_documents(documents)

        # Update instance chunks
        self.chunks = chunks

    def get_chunks(self):
        # If chunks is None, split the document and update chunks
        """
        Return the chunks of the document.

        Returns:
            A tuple containing the chunks as a list and the number of chunks as an int.
        """
        if self.chunks is None:
            self._split_document()

        logging.info(f"Total number of chunks: {len(self.chunks)}")
        return self.chunks, len(self.chunks)

    def reset(self):
        """Reset everything."""
        self.knowledge_base = None
        self.chunks = None
        self.text_splitter = None
        self.document_loader = None
