import uuid
from typing import Type
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

class StoreConfig:
    def __init__(self,
                 use_embeddings: bool = True,
                 embeddings: Type[HuggingFaceEmbeddings] = HuggingFaceEmbeddings,
                 embedding_fields: list = [],
                 dims: int = 384):
        """
        Initialize the StoreConfig object with the given configuration.

        Args:
            use_embeddings (bool, optional): If True, use embeddings in the store. Defaults to True.
            embeddings (Type[HuggingFaceEmbeddings], optional): The embeddings model to use. Defaults to None.
            embedding_fields (list, optional): The fields to embed. Defaults to [].
            dims (int, optional): The dimensions of the embeddings. Defaults to 384.
        """
        self.use_embeddings = use_embeddings
        self.embeddings = embeddings
        self.embedding_fields = embedding_fields
        self.dims = dims

    def _build_index(self):
        """
        Build and return the index configuration for the store.

        Returns:
            The index configuration for the store.
        """
        if not self.use_embeddings:
            return None

        index = {
            "embed": self.embeddings,
            "dims": self.dims,
            "fields": self.embedding_fields
        }

        return index

    def _validate_user_id(self, user_id: str):
        """
        Validate that the user ID is a valid UUID string.

        If the user ID is not a valid UUID string, raise a ValueError with a message indicating that the user ID must be a valid UUID string.

        :raises ValueError: If the user ID is not a valid UUID string.
        """
        try:
            verified_uuid = uuid.UUID(user_id)

            # Update user_id
            self.user_id = str(verified_uuid)
        except ValueError:
            raise ValueError(
                f"User ID {user_id} must be a valid UUID string")

    def _prepare_search_params(self, config: RunnableConfig, user_query: str = None)->dict:     
        """
        Search in the store for relevant documents based on the given user query.

        Args:
            config (RunnableConfig): The configuration for the runnable.
            user_query (str, optional): The user's query to search. Defaults to None.

        Returns:
            list: A list of relevant documents in the store.
        """
        # self._validate_store()
        
        self.user_id = config['configurable']['user_id']
        
        # Validate user_id
        self._validate_user_id(self.user_id)

        # if not user_query:
        #     # searches = self.store.search((self.user_id, "history"), limit=2)
        #     search_params = {
        #         "namespace": (self.user_id, "history"),
        #         "limit": 2
        #     }
        #     return search_params
        # else:
        search_params = {
            "namespace": (self.user_id, "history"),
            "query": user_query,
            "limit": 2
        }
        # searches = self.store.search((self.user_id, "history"), query=user_query, limit=2)
        return search_params
    
    def _format_chat_history(self, history: list, store_type='InMemoeryStore'):
        """
        Format chat history into a list of query-bot pairs.

        Args:
            history (list): The list of chat history entries.

        Returns:
            list: The formatted list of query-bot pairs.
        """
        history_values = []
        if not history:
            # If history is empty
            return history_values
        for hist in history:
            if store_type == 'PostgresStore':
                values = hist.get('value')
            else:
                values = hist.dict().get("value")
            
            if values:
                history_values.append(HumanMessage(values['query']))
                history_values.append(AIMessage(values['bot']))

        return history_values