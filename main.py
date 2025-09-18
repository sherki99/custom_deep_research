import asyncio
from graph.workflow import create_workflow
from graph.state import OnlineResearchState

def run_online_research():
    """Main function to run the WebSearch workflow."""
    
    app = create_workflow()
    
    # Initial state
    initial_state = OnlineResearchState(
        research_topic="What is Vard(designer and shipbuilder) doing in AI",
        search_depth="deep",
        max_sources_per_query=5,
        max_total_sources=10,
        language="en",
        geographic_focus="uk",
        date_filter="month",
        source_types=["web", "news", "pdf"]  
    )


    print("Starting Online Research Workflow...")
    
    try:
        final_state = app.invoke(initial_state)
        print("Workflow completed!")
        return final_state
    except Exception as e:
        print(f"Workflow failed: {str(e)}")
        return None


if __name__ == "__main__":
     run_online_research()
