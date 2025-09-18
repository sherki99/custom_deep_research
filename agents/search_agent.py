import os
from dotenv import load_dotenv
from tools.websearch_serper_tool import create_websearch_tool
from pydantic import BaseModel
from graph.state import OnlineResearchState
from langchain_openai import AzureChatOpenAI
import json


load_dotenv(override=True)


def search_serper_node(state: OnlineResearchState) -> OnlineResearchState:
    """Node function that searches on serper complete""" 
    
    
    # llm = AzureChatOpenAI(
    #     azure_endpoint=os.getenv("AZURE_API_BASE"),
    #     api_key=os.getenv("AZURE_API_KEY"),
    #     api_version=os.getenv("AZURE_API_VERSION"),
    #     azure_deployment=os.getenv("LLM_DEPLOYMENT_NAME")     
    # )

    web_tool = create_websearch_tool()

    result_json = web_tool.func(
        research_topic=state.research_topic,
        search_type="search",
        num_results=state.max_sources_per_query,
        language=state.language
    )

    result = json.loads(result_json)
    state.raw_search_results = result.get("results", [])
    state.selected_urls = [item.get("url") for item in state.raw_search_results]
    
    print("The url of the best that we got from serper are ", state.selected_urls)
    

    # print(state.raw_search_results)
    state.current_step = "web_search_completed"
    return state



