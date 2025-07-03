import time
from typing_extensions import Type
from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy import and_, between
from langchain_core.runnables import RunnableConfig
from langgraph.store.postgres import PostgresStore
from . import StoreConfig


class PostgresStoreConfig(StoreConfig):
    def __init__(self, use_embeddings = True, embeddings = None, embedding_fields = [], dims = 384):
        """
        Initialize a PostgresStoreConfig object with the given configuration.

        Args:
            use_embeddings (bool, optional): If True, it will use embeddings in the store for similarity search. Defaults to True.
            embeddings (Type[HuggingFaceEmbeddings], optional): The embeddings model to use. Defaults to None.
            embedding_fields (list, optional): The fields to embed. Defaults to [].
            dims (int, optional): The dimensions of the embeddings. Defaults to 384.
        """
        super().__init__(use_embeddings, embeddings, embedding_fields, dims)

        # Attach index from parent class *StoreConfig* to this class *PostgresStoreConfig*
        self.index = self._build_index()
        print("\n-----------------\nIndex: ", self.index, end='\n-------------------\n')
        self.store_type = PostgresStore
   
    def set_connection_string(self, user: str, password: str, host: str, port: int, database: str):
        """
        Set the connection string to connect to the Postgres database.

        Args:
            user (str): The username for the Postgres database.
            password (str): The password for the Postgres database.
            host (str): The hostname or IP address of the Postgres database.
            port (int): The port number for the Postgres database.
            database (str): The name of the database.
        """
        self.conn_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    def initial_postgres_store_setup(self):
        with self.store_type.from_conn_string(self.conn_string, index=self.index) as store:
            store.setup()

    def _search_in_store(self, config, user_query=None, is_latest=False):
        """
        Search in the store for relevant documents based on the given user query.

        Args:
            config (RunnableConfig): The configuration for the runnable.
            user_query (str, optional): The user's query to search. Defaults to None.

        Returns:
            list: A list of relevant documents in the store.
        """
        # Initial db setup if not done
        self.initial_postgres_store_setup()

        search_params = self._prepare_search_params(config, user_query)
        namespace = search_params.get('namespace', None)
        if not namespace:
            namespace = (self.user_id, "history")
        del search_params['namespace']
        
        # Check if connection string is set
        if not self.conn_string:
            raise ValueError("Connection string is not set. Please set it using set_connection_string method.")

        # Limit search to only consider today's history
        date = datetime.now()
        start_of_today = datetime.combine(date.today(), datetime.min.time(), tzinfo=ZoneInfo("Africa/Kigali"))
        end_of_today = datetime.combine(date.today(), datetime.max.time(), tzinfo=ZoneInfo("Africa/Kigali"))
        
        query_embedding = self.embeddings.embed_query(user_query)
        with self.store_type.from_conn_string(self.conn_string, index=self.index) as store:
            if is_latest:
                try:
                    query = """SELECT prefix, key, value, created_at, updated_at FROM store where prefix = %s ORDER BY created_at DESC LIMIT 2"""
                    emb_data = store.conn.execute(query, (f"{self.user_id}.history",)).fetchall()
                except Exception as e:
                    print("Error ---: ", e)
                    raise
            else:
                query = """
                    SELECT s.key, s.value, sv.created_at, sv.updated_at,
                        (sv.embedding <=> %s::vector) AS distance
                    FROM store_vectors AS sv
                    INNER JOIN store AS s ON s.prefix = sv.prefix
                    WHERE sv.prefix = %s
                    AND sv.created_at BETWEEN %s AND %s
                    ORDER BY sv.embedding <=> %s::vector
                    LIMIT %s
                """
                emb_data = store.conn.execute(
                    query,
                    (
                        query_embedding,  # once for distance calculation
                        f"{self.user_id}.history",
                        start_of_today,
                        end_of_today,
                        query_embedding,  # again for ORDER BY
                        search_params.get('limit'),
                    )
                ).fetchall()

            return emb_data
        
    def _get_latest_chat(self, user_query: str, config: RunnableConfig):
        """
        Retrieve the latest chat from the store.

        Returns:
            The most recent chat entry in the store.
        """
        store_data = self._search_in_store(config, user_query, is_latest=True)
        if not store_data:
            return []
        last_chat = store_data[0] #Refer to the first element which is the latest

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
            formatted = super()._format_chat_history(history, store_type="PostgresStore")
        else:
            latest = self._get_latest_chat(user_query, config)
            if latest:
                formatted = super()._format_chat_history([latest], store_type="PostgresStore")
            else:
                formatted = []

        return formatted

    def update_chat_history(self, data: dict, index_keys: list = ["query", "bot"]):
        """
        Update the store with the latest chat data.

        Args:
            data (dict): The query and bot response to store.
            index_keys (list, optional): The keys to index in the store. Defaults to ["query", "bot"].
        """
        namespace = (self.user_id, "history")
        unique_key = f"chat_{int(time.time() * 1000)}"
        with self.store_type.from_conn_string(self.conn_string, index=self.index) as store:
            store.put(namespace, unique_key, {f"{index_keys[0]}": data["query"], f"{index_keys[1]}": data["bot"]}, index=self.index.get('fields'))