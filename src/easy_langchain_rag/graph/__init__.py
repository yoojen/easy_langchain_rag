from typing_extensions import List, Dict, Tuple, Union, Type
from pydantic import BaseModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.redis import RedisSaver
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from langgraph.store.base import BaseStore


class GraphBuilder:
    def __init__(
        self,
        state: Type[Union[Dict, BaseModel]],
        nodes: Type[List[Tuple[str, callable]]],
        check_pointer: Type[Union[InMemorySaver, MemorySaver, PostgresSaver, RedisSaver]],
        store: Type[BaseStore],
        entry_point: str = None,
        tools: Type[List[BaseTool]] = ToolNode([])
    ):
        """
        Initialize a Graph object.

        Args:
            state (Type[Union[Dict, BaseModel]]): The initial state of the graph.
            nodes (Type[List[Tuple[str, callable]]]): A list of node identifiers and their corresponding functions.
            check_pointer (Type[Union[InMemorySaver, MemorySaver, PostgresSaver, RedisSaver]]): The checkpointer for saving graph states.
            store (Type[BaseStore]): The storage backend for persisting graph data.
            entry_point (str): The entry point node for the graph.
            tools (Type[List[BaseTool]], optional): A list of tools to be used within the graph. Defaults to an empty ToolNode list.
        """
        self.state = state
        self.nodes = nodes
        self.tools = tools
        self.check_pointer = check_pointer
        self.store = store
        self.entry_point = entry_point

    def _initialize_state(self):
        """
        Initialize a StateGraph with a given state.

        Returns:
            StateGraph: The initialized StateGraph.
        """
        graph_builder = StateGraph(self.state)
        return graph_builder

    def _construct_nodes(self):
        """
        Construct nodes in the graph.

        Iterate over the list of nodes and add them to the graph builder. If tools are provided, add a tools node to the graph. Finally, bind the graph builder to the class.

        :return: None
        """
        graph_builder = self._initialize_state()

        for node in self.nodes:
            graph_builder.add_node(node[0], node[1])

        # Add tools node if provided
        if self.tools:
            graph_builder.add_node("tools", self.tools)

        # Runtime binding graph_builder to the class
        self.graph_builder = graph_builder

    def _consturct_edges(self, tools_condition: callable = None):
        # Connect entry_point
        """
        Construct edges in the graph.

        Connect the entry point to the first tool or the start of the graph. Then connect all the nodes in order. Finally, connect the last node to the end of the graph.

        :param tools_condition: The condition to use for the tools node.
        :return: The constructed graph builder
        """

        try:
            if not self.entry_point:
                raise ValueError("Entry point not specified")
            self.graph_builder.set_entry_point(self.entry_point)

            # Add edges
            normal_edges = []
            for node in self.nodes:
                if node[0] == self.entry_point:
                    self.graph_builder.add_conditional_edges(
                        self.entry_point,
                        tools_condition,
                        {
                            END: END,
                            "tools": "tools"
                        },
                    )
                else:
                    normal_edges.append(node)
            normal_edges = [node for node in self.nodes if node[0] != self.entry_point]
            
            if not normal_edges:
                # Only entry point exists, connect it directly to END
                self.graph_builder.add_edge(self.entry_point, END) 

            for i in range(len(normal_edges)):
                # Last edge
                if i == 0 and len(normal_edges) > 1:
                    self.graph_builder.add_edge("tools", normal_edges[i][0])
                    continue
                
                elif i == 0 and len(normal_edges) == 1:
                    self.graph_builder.add_edge("tools", normal_edges[i][0])
                    self.graph_builder.add_edge(normal_edges[i][0], END)
                    break

                elif i == len(normal_edges) - 1: # If it is the last in the list
                    self.graph_builder.add_edge(normal_edges[i-1][0], normal_edges[i][0])
                    self.graph_builder.add_edge(normal_edges[i][0], END)
                    break
                else:
                    # All other edges
                    self.graph_builder.add_edge(normal_edges[i-1][0], normal_edges[i][0])
            
            return self.graph_builder
        except Exception as e:
            print("\n\nError: ", e, "\n\n")
            raise

    def compile_graph(self, tools_condition: callable = None):
        """
        Compile the graph by constructing nodes and edges.

        It first constructs the nodes using the given nodes and tools. Then it constructs the edges by connecting all the nodes in order. Finally, it compiles the graph with the given checkpointer and store.

        Returns:
            The compiled graph.
        """
        try:
            self._construct_nodes()

            # Construct edges
            graph_builder = self._consturct_edges(tools_condition)
            # Compile
            graph = graph_builder.compile(checkpointer=self.check_pointer, store=self.store)
            return graph
        except Exception as e:
            print("\n\nError: ", e, "\n\n")
            raise