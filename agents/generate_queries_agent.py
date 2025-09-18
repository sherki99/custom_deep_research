import os 
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from graph.state import OnlineResearchState



def generate_queries_llm_node(state: OnlineResearchState) -> OnlineResearchState: 
    """Use llm to generate search queries based on the topic and stategy"""

    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_deployment=os.getenv("LLM_DEPLOYMENT_NAME")
    )

    # Prompt to instruct the LLM
    prompt = (
        f"Generate exactly 5 distinct search queries for the topic: '{state.research_topic}'.\n"
        f"Use the strategy: '{state.query_strategy}'.\n\n"
        "Requirements:\n"
        "- Only return the 5 queries, one per line.\n"
        "- Do not include numbering, bullet points, or extra explanations.\n"
        "- Each query should be short, clear, and specific.\n"
    )



    response = llm.invoke([{"role": "user", "content": prompt}])
    queries_text = response.content

    queries =  [q.strip("- ") for q in queries_text.split("\n") if q.strip()]
    if not queries[0][0].isdigit():
        queries = queries[1:]

    
    state.search_queries =  queries
    print("First step is to Generate queries: ", state.search_queries)

    return state
