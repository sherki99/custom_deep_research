import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
import requests

load_dotenv(override=True)

class WebSearchInput(BaseModel):
    """Input schema for WebSearchTool with more customization"""
    research_topic: str = Field(..., description="Search research_topic to run on the web/news/scholar")
    search_type: str = Field(
        default="search",
        description="Type of search: 'search' (web), 'news' (Google News), or 'scholar' (Google Scholar)"
    )
    num_results: int = Field(default=5, description="Max number of results to return")
    location: Optional[str] = Field(None, description="Location to bias results (e.g., 'New York, USA')")
    country: Optional[str] = Field(None, description="Country code (e.g., 'us', 'de', 'in')")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'de', 'fr')")
    time_range: Optional[str] = Field(
        None,
        description="Time filter for results: 'd' (day), 'w' (week), 'm' (month)"
    )
    safe_search: bool = Field(default=False, description="Enable safe search to filter explicit results")


def web_search_function(
    research_topic: str,
    search_type: str = "search",
    num_results: int = 5,
    location: Optional[str] = None,
    country: Optional[str] = None,
    language: Optional[str] = None,
    time_range: Optional[str] = None,
    safe_search: bool = False,
) -> str:
    """
    Search the web, news, or scholar using Serper API with customization.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return json.dumps({"error": "SERPER_API_KEY not set", "results": []})

    url = f"https://google.serper.dev/{search_type}"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    payload = {"q": research_topic, "num": num_results}
    if location: payload["location"] = location
    if country: payload["gl"] = country
    if language: payload["hl"] = language
    if time_range: payload["tbs"] = f"qdr:{time_range}"
    if safe_search: payload["safe"] = "active"

    try:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()


        results = []
        for item in data.get("organic", [])[:num_results]:
           # print(item)
            results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "url": item.get("link"),
                "date": item.get("date"),
                "source": item.get("source") or item.get("domain") or "UnKnow",
            })

        return json.dumps({
            "query": research_topic,
            "search_type": search_type,
            "total_results": len(results),
            "results": results,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "results": []})

def create_websearch_tool():
    """Create LangChain StructuredTool for Serper web search"""
    return StructuredTool.from_function(
        name="web_search",
        description="Search the web/news/scholar using the Serper API",
        func=web_search_function,
        args_schema=WebSearchInput,
        coroutine=None  # Not async anymore
    )
