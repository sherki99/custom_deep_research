import os
import json
from dotenv import load_dotenv
from tools.analysis_synthesis_tool import create_analysis_synthesis_tool
from graph.state import OnlineResearchState

load_dotenv(override=True)

def analysis_synthesis_node(state: OnlineResearchState) -> OnlineResearchState:
    """Node function that analyzes and synthesizes extracted content"""
    
    print("Starting content analysis and synthesis...")
    
    # Check if we have extracted content to analyze
    if not state.extracted_content:
        print("No extracted content to analyze")
        state.current_step = "analysis_synthesis_skipped"
        state.errors.append("No extracted content available for analysis")
        return state
    
    # Check if there are any successful extractions
    successful_extractions = {
        url: content for url, content in state.extracted_content.items() 
        if 'error' not in content
    }
    
    if not successful_extractions:
        print("No successful content extractions to analyze")
        state.current_step = "analysis_synthesis_failed"
        state.errors.append("No successful content extractions available for analysis")
        return state
    
    try:
        # Create the analysis synthesis tool
        analysis_tool = create_analysis_synthesis_tool()
        
        # Prepare analysis parameters based on state
        analysis_depth = "comprehensive"  # Default
        if state.search_depth == "shallow":
            analysis_depth = "standard"
        elif state.search_depth == "deep":
            analysis_depth = "deep"
        
        # Call the analysis tool
        result_json = analysis_tool.func(
            research_topic=state.research_topic,
            extracted_content=state.extracted_content,
            search_queries=state.search_queries,
            analysis_depth=analysis_depth,
            focus_areas=[],  # Can be made configurable in state if needed
            include_contradictions=True,
            include_gaps=True,
            max_findings=10
        )
        
        # Parse the results
        result = json.loads(result_json)
        
        # Check for errors in the result
        if 'error' in result:
            print(f"Analysis failed: {result['error']}")
            state.current_step = "analysis_synthesis_failed"
            state.errors.append(f"Analysis failed: {result['error']}")
            return state
        
        # Update state with analysis results
        state.key_findings = result.get("key_findings", [])
        state.contradictions = result.get("contradictions", [])
        state.consensus_points = result.get("consensus_points", [])
        state.research_gaps = result.get("research_gaps", [])
        state.executive_summary = result.get("executive_summary", "")
        
        # Update quality metrics
        quality_metrics = result.get("quality_metrics", {})
        state.source_diversity_score = quality_metrics.get("source_diversity_score", 0.0)
        state.information_depth_score = quality_metrics.get("information_depth_score", 0.0)
        state.credibility_average = quality_metrics.get("credibility_average", 0.0)
        state.coverage_completeness = quality_metrics.get("coverage_completeness", 0.0)
        
        # Update current step
        state.current_step = "analysis_synthesis_completed"
        
        # Log results
        analysis_stats = result.get("analysis_stats", {})
        print(f"Analysis and synthesis completed:")
        print(f"  - Sources analyzed: {analysis_stats.get('sources_analyzed', 0)}")
        print(f"  - Total word count: {analysis_stats.get('total_word_count', 0):,}")
        print(f"  - Key findings extracted: {analysis_stats.get('findings_extracted', 0)}")
        print(f"  - Contradictions found: {analysis_stats.get('contradictions_found', 0)}")
        print(f"  - Consensus points identified: {analysis_stats.get('consensus_points_found', 0)}")
        print(f"  - Research gaps identified: {analysis_stats.get('research_gaps_identified', 0)}")
        
        # Log quality metrics
        print(f"Quality Metrics:")
        print(f"  - Source diversity: {state.source_diversity_score:.2f}")
        print(f"  - Information depth: {state.information_depth_score:.2f}")
        print(f"  - Credibility average: {state.credibility_average:.2f}")
        print(f"  - Coverage completeness: {state.coverage_completeness:.2f}")
        
        # Show top findings
        if state.key_findings:
            print("Top Key Findings:")
            for i, finding in enumerate(state.key_findings[:3], 1):
                confidence = finding.get('confidence', 0)
                category = finding.get('category', 'general')
                finding_text = finding.get('finding', '')[:100] + "..." if len(finding.get('finding', '')) > 100 else finding.get('finding', '')
                print(f"  {i}. [{category.upper()}] {finding_text} (Confidence: {confidence}/5)")
        
        # Show contradictions if any
        if state.contradictions:
            print(f"Contradictions found: {len(state.contradictions)}")
            for i, contradiction in enumerate(state.contradictions[:2], 1):
                contradiction_text = contradiction.get('contradiction', '')[:80] + "..." if len(contradiction.get('contradiction', '')) > 80 else contradiction.get('contradiction', '')
                severity = contradiction.get('severity', 'unknown')
                print(f"  {i}. [{severity.upper()}] {contradiction_text}")
        
        # Show consensus points if any
        if state.consensus_points:
            print(f"Consensus points identified: {len(state.consensus_points)}")
            for i, consensus in enumerate(state.consensus_points[:2], 1):
                consensus_text = consensus.get('consensus_point', '')[:80] + "..." if len(consensus.get('consensus_point', '')) > 80 else consensus.get('consensus_point', '')
                strength = consensus.get('strength', 0)
                print(f"  {i}. {consensus_text} (Strength: {strength}/5)")
        
        # Show research gaps if any
        if state.research_gaps:
            print(f"Research gaps identified: {len(state.research_gaps)}")
            for i, gap in enumerate(state.research_gaps[:2], 1):
                gap_text = gap[:80] + "..." if len(gap) > 80 else gap
                print(f"  {i}. {gap_text}")
        
        return state
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse analysis results: {str(e)}"
        print(error_msg)
        state.current_step = "analysis_synthesis_failed"
        state.errors.append(error_msg)
        return state
        
    except Exception as e:
        error_msg = f"Analysis and synthesis failed: {str(e)}"
        print(error_msg)
        state.current_step = "analysis_synthesis_failed"
        state.errors.append(error_msg)
        return state