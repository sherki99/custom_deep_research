import os
from dotenv import load_dotenv
from tools.content_extraction_bs4_tool import create_content_extraction_tool
from pydantic import BaseModel
from graph.state import OnlineResearchState
from langchain_openai import AzureChatOpenAI
import json

load_dotenv(override=True)

def extract_content_node(state: OnlineResearchState) -> OnlineResearchState:
    """Node function that extracts content from validated URLs"""
    
    print("Starting content extraction...")
    
    # Check if we have URLs to process
    if not state.selected_urls:
        print("No URLs to extract content from")
        state.current_step = "content_extraction_skipped"
        return state
    
    # Create the content extraction tool
    extraction_tool = create_content_extraction_tool()
    
    # Call the extraction tool with validated URLs
    result_json = extraction_tool.func(
        urls=state.selected_urls,
        max_content_length=10000,  # You can make this configurable in state if needed
        timeout=15,
        extract_metadata=True,
        extract_links=False,  # Set to True if you want to extract links
        delay_between_requests=1.0  # Be respectful to servers
    )
    
    # Parse the results
    result = json.loads(result_json)
    
    # Update state with extraction results
    state.extracted_content = result.get("extracted_content", {})
    state.content_stats = result.get("content_stats", {})
    state.failed_extractions = result.get("failed_extractions", [])
    
    # Update current step
    state.current_step = "content_extraction_completed"
    
    # Log results
    stats = state.content_stats
    print(f"Content extraction completed:")
    print(f"  - Total URLs processed: {stats.get('total_urls', 0)}")
    print(f"  - Successful extractions: {stats.get('successful_extractions', 0)}")
    print(f"  - Failed extractions: {stats.get('failed_extractions', 0)}")
    print(f"  - Total words extracted: {stats.get('total_word_count', 0):,}")
    print(f"  - Average content length: {stats.get('average_content_length', 0):.0f} chars")
    
    # Show successful extractions
    successful_count = 0
    for url, content in state.extracted_content.items():
        if 'error' not in content:
            successful_count += 1
            title = content.get('title', 'No title')
            word_count = content.get('stats', {}).get('word_count', 0)
            print(f"  {successful_count}. {title[:60]}... ({word_count} words)")
            
            if successful_count >= 3:  # Show first 3
                break
    
    # Show failed extractions if any
    if state.failed_extractions:
        print("Failed extractions:")
        for i, failed_url in enumerate(state.failed_extractions[:3], 1):
            print(f"  {i}. {failed_url}")
            # Show error reason if available
            if failed_url in state.extracted_content:
                error = state.extracted_content[failed_url].get('error', 'Unknown error')
                print(f"     Error: {error}")
    
    return state