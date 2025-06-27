from typing_extensions import Type
from contextlib import contextmanager
from langchain_core.runnables import RunnableConfig
from langgraph.store.memory import InMemoryStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from . import StoreConfig


class InMemoryStoreConfig(StoreConfig):
    def __init__(self, store_type = InMemoryStore, use_embeddings = True, embeddings = None, embedding_fields = [], dims = 384):
        if store_type != InMemoryStore:
            raise ValueError("InMemoryStoreConfig must use InMemoryStore as the store type.")
        
        # Initialize the base StoreConfig
        super().__init__(use_embeddings, embeddings, embedding_fields, dims)
        
        # Create the store with the built index
        self.index = self._build_index()
        print("\nIndex: ", self.index)
        self.store = store_type(index=self.index)

    def _search_in_store(self, config: RunnableConfig, user_query: str = None)->list:
        """
        Search in the store for relevant documents based on the given user query.

        Args:
            config (RunnableConfig): The configuration for the runnable.
            user_query (str, optional): The user's query to search. Defaults to None.

        Returns:
            list: A list of relevant documents in the store.
        """
        # self._validate_store()
        
        # self.user_id = config['configurable']['user_id']
        
        # # Validate user_id
        # self._validate_user_id()
        search_params = self._prepare_search_params(config, user_query)
        namespace = search_params.get('namespace', None)
        if not namespace:
            namespace = (self.user_id, "history") # Fallback to default namespace
        del search_params['namespace']

        # Search in the store
        searches = self.store.search(namespace, **search_params)
        return searches
    
    def _get_latest_chat(self, user_query: str, config: RunnableConfig):
        """
        Retrieve the latest chat from the store.

        Returns:
            The most recent chat entry in the store.
        """
        store_data = self._search_in_store(config, user_query)
        if not store_data:
            return []
        last_chat = store_data[-1]

        return last_chat
    
    def load_chat_history(self, user_query:str, config: RunnableConfig, is_latest=False):
        """
        Load chat history from the store.

        Args:
            is_latest (bool, optional): If True, only load the latest chat entry. Defaults to False.

        Returns:
            list: The formatted chat history.
        """
        if not is_latest:
            history = self._search_in_store(config, user_query)
            formatted = super()._format_chat_history(history)
        else:
            latest = self._get_latest_chat(user_query, config)
            if latest:
                formatted = super()._format_chat_history([latest])
            else:
                formatted = []

        return formatted

    def update_chat_history(self, data: dict, index_keys: list = ["query", "bot"], index: list = ["query", "bot"]):
        """
        Update the store with the latest chat data.

        Args:
            data (dict): The query and bot response to store.
        """
        namespace = (self.user_id, "history")
        # memory_id = str(uuid.uuid4())

        # We create a new memory
        self.store.put(namespace, "chat_history", {f"{index_keys[0]}": data["query"], f"{index_keys[1]}": data["bot"]}, index=index)