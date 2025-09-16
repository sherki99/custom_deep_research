from graph.workflow import create_workflow
from graph.state import OnlineResearchState
import asyncio

async def run_online_research():
    """Main function to run the WebSearch workflow."""
    
    app = create_workflow()
    
    # Initial state
    initial_state = OnlineResearchState(
        research_topic="How to use cursor AI",
        max_sources_per_query=5,
        max_total_sources=10,
        language="en"
    )
    
    print("Starting Online Research Workflow...")
    
    try:
        final_state =  await app.ainvoke(initial_state)
        print("Workflow completed!")
        return final_state
    except Exception as e:
        print(f"Workflow failed: {str(e)}")
        return None


if __name__ == "__main__":
    
    asyncio.run(run_online_research())
