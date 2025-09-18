import os
from dotenv import load_dotenv
from tools.source_validation_tool import create_source_validation_tool
from pydantic import BaseModel
from graph.state import OnlineResearchState
from langchain_openai import AzureChatOpenAI
import json

load_dotenv(override=True)

def validate_sources_node(state: OnlineResearchState) -> OnlineResearchState:
    """Node function that validates sources and assigns credibility scores"""
    
    print("Starting source validation...")
    
    # Create the validation tool
    validation_tool = create_source_validation_tool()
    
    # Call the validation tool with current search results
    result_json = validation_tool.func(
        search_results=state.raw_search_results,
        min_credibility_threshold=0.3,  # You can make this configurable in state if needed
        llm_weight=0.4,  # 40% weight for LLM assessment as requested
        check_accessibility=True,
        timeout=10
    )
    
    # Parse the results
    result = json.loads(result_json)
    
    # Update state with validation results
    state.validated_sources = result.get("validated_sources", {})
    state.source_credibility_scores = result.get("credibility_scores", {})
    state.removed_sources = result.get("removed_sources", [])
    state.credibility_average = result.get("credibility_average", 0.0)
    
    # Update selected URLs to only include validated sources
    state.selected_urls = list(state.validated_sources.keys())
    
    # Update current step
    state.current_step = "source_validation_completed"
    
    # Log results
    print(f"Source validation completed:")
    print(f"  - Total validated sources: {result.get('total_validated', 0)}")
    print(f"  - Total removed sources: {result.get('total_removed', 0)}")
    print(f"  - Average credibility: {state.credibility_average:.2f}")
    
    # Show validated URLs
    print("Validated URLs:")
    for i, url in enumerate(state.selected_urls[:5], 1):  # Show first 5
        score = state.source_credibility_scores.get(url, 0.0)
        print(f"  {i}. {url} (Score: {score:.2f})")
    
    # Show removed sources (first 3)
    if state.removed_sources:
        print("Removed sources:")
        for i, removed in enumerate(state.removed_sources[:3], 1):
            print(f"  {i}. {removed.get('title', 'Unknown')[:50]}... ({removed.get('reason', 'Unknown reason')})")
    
    return state


