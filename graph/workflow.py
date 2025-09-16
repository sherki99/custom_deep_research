from langgraph.graph import StateGraph, END
from graph.state import OnlineResearchState
from agents.search_agent import search_serper_node


def create_workflow():
    """create the workflow for online research using websearch"""

    workflow =  StateGraph(OnlineResearchState)

    workflow.add_node("web_search", search_serper_node)

    workflow.set_entry_point("web_search")

    workflow.add_edge("web_search", END)

    return workflow.compile()

