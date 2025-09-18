import os
import json
from dotenv import load_dotenv
from tools.report_generation_tool import create_report_generation_tool
from graph.state import OnlineResearchState
from datetime import datetime

load_dotenv(override=True)

def report_generation_node(state: OnlineResearchState) -> OnlineResearchState:
    """Node function that generates final research reports"""
    
    print("Starting report generation...")
    
    # Check if we have analysis results to generate report from
    if not state.key_findings and not state.consensus_points and not state.contradictions:
        print("No analysis results available for report generation")
        state.current_step = "report_generation_skipped"
        state.warnings.append("No analysis results available for report generation")
        
        # Generate a basic report if we have extracted content
        if state.extracted_content:
            basic_report = generate_basic_report(state)
            state.synthesized_report = basic_report
            state.detailed_analysis = basic_report
            
        
        return state
    
    try:
        # Create the report generation tool
        report_tool = create_report_generation_tool()
        
        # Prepare quality metrics dictionary
        quality_metrics = {
            'source_diversity_score': state.source_diversity_score,
            'information_depth_score': state.information_depth_score,
            'credibility_average': state.credibility_average,
            'coverage_completeness': state.coverage_completeness
        }
        
        # Determine report type based on search depth
        report_type = "standard"
        if state.search_depth == "shallow":
            report_type = "brief"
        elif state.search_depth == "deep":
            report_type = "comprehensive"
        
        # Call the report generation tool
        result_json = report_tool.func(
            research_topic=state.research_topic,
            key_findings=state.key_findings,
            contradictions=state.contradictions,
            consensus_points=state.consensus_points,
            research_gaps=state.research_gaps,
            executive_summary=state.executive_summary,
            extracted_content=state.extracted_content,
            quality_metrics=quality_metrics,
            report_type=report_type,
            include_methodology=True,
            include_bibliography=True,
            target_audience="general"
        )
        
        # Parse the results
        result = json.loads(result_json)
        
        # Update state with report results
        state.synthesized_report = result.get("synthesized_report", "")
        state.detailed_analysis = result.get("detailed_analysis", "")
        state.source_bibliography = result.get("source_bibliography", [])
        
        # If executive summary wasn't generated before, use the one from report
        if not state.executive_summary:
            state.executive_summary = result.get("executive_summary", "")
        
        # Update current step
        state.current_step = "report_generation_completed"
        save_report_md(state)
        
        # Log results
        report_metadata = result.get("report_metadata", {})
        statistics = report_metadata.get("statistics", {})
        
        print(f"Report generation completed:")
        print(f"  - Report type: {report_metadata.get('report_type', 'unknown')}")
        print(f"  - Total words: {statistics.get('total_report_words', 0):,}")
        print(f"  - Sources cited: {statistics.get('sources_cited', 0)}")
        print(f"  - Findings included: {statistics.get('findings_included', 0)}")
        print(f"  - Contradictions discussed: {statistics.get('contradictions_discussed', 0)}")
        print(f"  - Consensus points highlighted: {statistics.get('consensus_points_highlighted', 0)}")
        print(f"  - Research gaps identified: {statistics.get('research_gaps_identified', 0)}")
        
        # Show sections included
        sections = report_metadata.get("sections_generated", {})
        sections_list = []
        if sections.get("main_report"):
            sections_list.append("Main Report")
        if sections.get("methodology"):
            sections_list.append("Methodology")
        if sections.get("bibliography"):
            sections_list.append("Bibliography")
        
        if sections_list:
            print(f"  - Sections included: {', '.join(sections_list)}")
        
        # Show preview of executive summary
        if state.executive_summary:
            preview = state.executive_summary[:200] + "..." if len(state.executive_summary) > 200 else state.executive_summary
            print(f"Executive Summary Preview: {preview}")
        
        return state
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse report generation results: {str(e)}"
        print(error_msg)
        state.current_step = "report_generation_failed"
        state.errors.append(error_msg)
        
        # Try to generate a basic report as fallback
        if state.extracted_content:
            basic_report = generate_basic_report(state)
            state.synthesized_report = basic_report
            state.detailed_analysis = basic_report
        
        return state
        
    except Exception as e:
        error_msg = f"Report generation failed: {str(e)}"
        print(error_msg)
        state.current_step = "report_generation_failed"
        state.errors.append(error_msg)
        
        # Try to generate a basic report as fallback
        if state.extracted_content:
            basic_report = generate_basic_report(state)
            state.synthesized_report = basic_report
            state.detailed_analysis = basic_report
        
        return state

def generate_basic_report(state: OnlineResearchState) -> str:
    """Generate a basic report when full report generation fails"""
    
    from datetime import datetime
    
    # Count successful extractions
    successful_sources = [
        (url, content) for url, content in state.extracted_content.items() 
        if 'error' not in content
    ]
    
    # Basic report template
    report = f"""# Research Report: {state.research_topic}

## Executive Summary

This research was conducted on "{state.research_topic}" using automated content analysis. 
{len(successful_sources)} sources were successfully analyzed from {len(state.extracted_content)} total sources attempted.

## Key Information Gathered

"""
    
    # Add source summaries
    for i, (url, content) in enumerate(successful_sources[:5], 1):
        title = content.get('title', 'Untitled')
        description = content.get('description', '')
        word_count = content.get('stats', {}).get('word_count', 0)
        
        report += f"### Source {i}: {title}\n"
        report += f"- **URL**: {url}\n"
        report += f"- **Content Length**: {word_count:,} words\n"
        if description:
            report += f"- **Description**: {description}\n"
        report += "\n"
    
    # Add analysis results if available
    if state.key_findings:
        report += "## Key Findings\n\n"
        for i, finding in enumerate(state.key_findings[:5], 1):
            report += f"{i}. {finding.get('finding', 'No finding text')}\n"
        report += "\n"
    
    if state.research_gaps:
        report += "## Research Gaps Identified\n\n"
        for i, gap in enumerate(state.research_gaps[:3], 1):
            report += f"{i}. {gap}\n"
        report += "\n"
    
    # Add metadata
    report += f"""## Research Metadata

- **Research Topic**: {state.research_topic}
- **Sources Analyzed**: {len(successful_sources)}
- **Total Sources Attempted**: {len(state.extracted_content)}
- **Search Depth**: {state.search_depth}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

*This is a basic report generated due to limitations in the full report generation process.*
"""
    
    return report



def save_report_md(state: OnlineResearchState, filename: str = f"research_report{datetime.date}.md") -> None: 
    """Save the synthsized or basic report to a file"""

    try:
        report_content = state.synthesized_report or state.detailed_analysis
        if not report_content: 
            print("No report available to save.")
            return 
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"Report succesfully saved to {filename}")

    except Exception as e:
        print(f"failed to save report {e}")
        state.errors.append(f"Report save failed: {str(e)}")