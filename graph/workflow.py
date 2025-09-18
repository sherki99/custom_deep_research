from langgraph.graph import StateGraph, END
from graph.state import OnlineResearchState
from agents.search_agent import search_serper_node
from agents.generate_queries_agent import generate_queries_llm_node
from agents.validate_sources_agent import validate_sources_node
from agents.content_extraction import extract_content_node
from agents.analysis_systhesis_agent import analysis_synthesis_node
from agents.report_generation_agent import report_generation_node


def create_workflow():
    """Create the workflow for online research using websearch"""
    
    workflow = StateGraph(OnlineResearchState)
    
    # Add all nodes
    workflow.add_node("generate_queries", generate_queries_llm_node)
    workflow.add_node("web_search", search_serper_node)
    workflow.add_node("validate_sources", validate_sources_node)
    workflow.add_node("content_extraction", extract_content_node)
    workflow.add_node("analysis_synthesis", analysis_synthesis_node)
    workflow.add_node("report_generation", report_generation_node)
    
    # Set entry point
    workflow.set_entry_point("generate_queries")
    
    # Define the flow
    workflow.add_edge("generate_queries", "web_search")
    workflow.add_edge("web_search", "validate_sources")
    workflow.add_edge("validate_sources", "content_extraction")
    workflow.add_edge("content_extraction", "analysis_synthesis")
    workflow.add_edge("analysis_synthesis", "report_generation")
    workflow.add_edge("report_generation", END)
    
    return workflow.compile()