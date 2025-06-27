from langchain.callbacks.base import BaseCallbackHandler
from queue import Queue, Empty
from typing_extensions import Dict, Any, Optional


class SSEHandler(BaseCallbackHandler):
    def __init__(self, queue: Queue):
        self.queue = queue
        self.should_stream = False

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Track the current node name when a chain starts."""
        langgraph_node = metadata.get("langgraph_node")

        if langgraph_node == "generate":
            self.should_stream = True
        else:
            self.should_stream = False

    def on_llm_new_token(self, token: str, **kwargs):
        if self.should_stream:
            self.queue.put(token)
            # print(self.queue.get(timeout=10), end="", flush=True)

    def on_llm_end(self, response, **kwargs):
        try:
            while True:
                self.queue.get_nowait()
        except Empty:
            pass
        self.should_stream = False
