import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv(override=True)

class ReportGenerationInput(BaseModel):
    """Input schema for Report Generation Tool"""
    research_topic: str = Field(..., description="The original research topic")
    key_findings: List[Dict[str, Any]] = Field(default=[], description="Key findings from analysis")
    contradictions: List[Dict[str, Any]] = Field(default=[], description="Contradictions found")
    consensus_points: List[Dict[str, Any]] = Field(default=[], description="Consensus points identified")
    research_gaps: List[str] = Field(default=[], description="Research gaps identified")
    executive_summary: str = Field(default="", description="Executive summary")
    extracted_content: Dict[str, Dict[str, Any]] = Field(default={}, description="Original extracted content")
    quality_metrics: Dict[str, float] = Field(default={}, description="Research quality metrics")
    report_type: str = Field(default="comprehensive", description="brief, standard, comprehensive, detailed")
    include_methodology: bool = Field(default=True, description="Include methodology section")
    include_bibliography: bool = Field(default=True, description="Include source bibliography")
    target_audience: str = Field(default="general", description="general, academic, executive, technical")

def report_generation_function(
    research_topic: str,
    key_findings: List[Dict[str, Any]] = None,
    contradictions: List[Dict[str, Any]] = None,
    consensus_points: List[Dict[str, Any]] = None,
    research_gaps: List[str] = None,
    executive_summary: str = "",
    extracted_content: Dict[str, Dict[str, Any]] = None,
    quality_metrics: Dict[str, float] = None,
    report_type: str = "comprehensive",
    include_methodology: bool = True,
    include_bibliography: bool = True,
    target_audience: str = "general"
) -> str:
    """
    Generate comprehensive research reports based on analysis results.
    """
    
    # Initialize default values
    if key_findings is None:
        key_findings = []
    if contradictions is None:
        contradictions = []
    if consensus_points is None:
        consensus_points = []
    if research_gaps is None:
        research_gaps = []
    if extracted_content is None:
        extracted_content = {}
    if quality_metrics is None:
        quality_metrics = {}
    
    # Initialize LLM
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_deployment=os.getenv("LLM_DEPLOYMENT_NAME"),
        temperature=0.2
    )
    
    def create_bibliography(content_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
        """Create a bibliography from extracted content"""
        
        bibliography = []
        for url, content_data in content_dict.items():
            if 'error' in content_data:
                continue
                
            title = content_data.get('title', 'Untitled')
            domain = url.split('/')[2] if '//' in url else url
            
            # Extract date if available
            metadata = content_data.get('metadata', {})
            date_published = metadata.get('published_date', '')
            if date_published:
                try:
                    # Try to parse and format date
                    if 'T' in date_published:
                        date_obj = datetime.fromisoformat(date_published.replace('Z', '+00:00'))
                        formatted_date = date_obj.strftime('%Y-%m-%d')
                    else:
                        formatted_date = date_published
                except:
                    formatted_date = date_published
            else:
                formatted_date = 'Date not available'
            
            author = metadata.get('author', domain)
            
            bibliography.append({
                'title': title,
                'author': author,
                'url': url,
                'domain': domain,
                'date_published': formatted_date,
                'date_accessed': datetime.now().strftime('%Y-%m-%d')
            })
        
        # Sort by domain and title
        bibliography.sort(key=lambda x: (x['domain'], x['title']))
        return bibliography
    
    def generate_detailed_report() -> str:
        """Generate a detailed research report"""
        
        report_prompt = ChatPromptTemplate.from_template("""
        You are an expert research analyst. Generate a comprehensive research report on "{research_topic}" based on the analysis results provided.
        
        Research Topic: {research_topic}
        Target Audience: {target_audience}
        Report Type: {report_type}
        
        Analysis Results:
        - Key Findings: {findings_count} findings identified
        - Contradictions: {contradictions_count} contradictions found
        - Consensus Points: {consensus_count} consensus points identified
        - Research Gaps: {gaps_count} gaps identified
        
        Key Findings:
        {key_findings_text}
        
        Contradictions:
        {contradictions_text}
        
        Consensus Points:
        {consensus_text}
        
        Research Gaps:
        {gaps_text}
        
        Quality Metrics:
        {quality_metrics_text}
        
        Please generate a comprehensive research report with the following structure:
        
        # Research Report: {research_topic}
        
        ## Executive Summary
        {executive_summary}
        
        ## Introduction
        [Brief introduction to the research topic and objectives]
        
        ## Key Findings
        [Detailed presentation of key findings with evidence and analysis]
        
        ## Areas of Agreement (Consensus)
        [Points where multiple sources agree, if any]
        
        ## Areas of Disagreement (Contradictions)
        [Conflicting information found, if any]
        
        ## Research Gaps and Future Directions
        [Areas needing further investigation]
        
        ## Data Quality Assessment
        [Assessment of source quality and research limitations]
        
        ## Conclusions
        [Summary of main conclusions and implications]
        
        Write in a professional, clear style appropriate for the {target_audience} audience. Use evidence from the findings to support all claims. Be objective and balanced in your analysis.
        """)
        
        # Prepare text summaries
        findings_text = "\n".join([
            f"- {finding.get('finding', '')} (Confidence: {finding.get('confidence', 0)}/5, Category: {finding.get('category', 'general')})"
            for finding in key_findings[:10]
        ]) if key_findings else "No key findings available."
        
        contradictions_text = "\n".join([
            f"- {contradiction.get('contradiction', '')} (Severity: {contradiction.get('severity', 'unknown')})"
            for contradiction in contradictions[:5]
        ]) if contradictions else "No significant contradictions found."
        
        consensus_text = "\n".join([
            f"- {consensus.get('consensus_point', '')} (Strength: {consensus.get('strength', 0)}/5)"
            for consensus in consensus_points[:5]
        ]) if consensus_points else "No strong consensus points identified."
        
        gaps_text = "\n".join([
            f"- {gap}" for gap in research_gaps[:8]
        ]) if research_gaps else "No specific research gaps identified."
        
        quality_text = "\n".join([
            f"- {key.replace('_', ' ').title()}: {value:.2f}"
            for key, value in quality_metrics.items()
        ]) if quality_metrics else "Quality metrics not available."
        
        try:
            response = llm.invoke(
                report_prompt.format(
                    research_topic=research_topic,
                    target_audience=target_audience,
                    report_type=report_type,
                    findings_count=len(key_findings),
                    contradictions_count=len(contradictions),
                    consensus_count=len(consensus_points),
                    gaps_count=len(research_gaps),
                    key_findings_text=findings_text,
                    contradictions_text=contradictions_text,
                    consensus_text=consensus_text,
                    gaps_text=gaps_text,
                    quality_metrics_text=quality_text,
                    executive_summary=executive_summary or "Executive summary not available."
                )
            )
            
            return response.content
            
        except Exception as e:
            return f"Error generating detailed report: {str(e)}"
    
    def generate_methodology_section() -> str:
        """Generate methodology section"""
        
        if not include_methodology:
            return ""
        
        methodology_prompt = ChatPromptTemplate.from_template("""
        Generate a methodology section for the research report on "{research_topic}".
        
        Research Details:
        - Sources analyzed: {sources_count}
        - Total content words: {total_words}
        - Source diversity score: {diversity_score}
        - Coverage completeness: {coverage_score}
        
        Include information about:
        1. Data collection methods
        2. Source selection criteria
        3. Content analysis approach
        4. Quality assessment methods
        5. Limitations of the methodology
        
        Write 2-3 paragraphs in a professional academic style.
        """)
        
        try:
            sources_count = len([url for url, content in extracted_content.items() if 'error' not in content])
            total_words = sum([
                content.get('stats', {}).get('word_count', 0) 
                for content in extracted_content.values() 
                if 'error' not in content
            ])
            
            response = llm.invoke(
                methodology_prompt.format(
                    research_topic=research_topic,
                    sources_count=sources_count,
                    total_words=total_words,
                    diversity_score=quality_metrics.get('source_diversity_score', 0.0),
                    coverage_score=quality_metrics.get('coverage_completeness', 0.0)
                )
            )
            
            return f"\n\n## Methodology\n{response.content}"
            
        except Exception as e:
            return f"\n\n## Methodology\nError generating methodology section: {str(e)}"
    
    def generate_bibliography_section(bibliography: List[Dict[str, str]]) -> str:
        """Generate bibliography section"""
        
        if not include_bibliography or not bibliography:
            return ""
        
        bib_text = "\n\n## Sources and References\n\n"
        
        for i, source in enumerate(bibliography, 1):
            title = source.get('title', 'Untitled')
            author = source.get('author', 'Unknown Author')
            url = source.get('url', '')
            date_published = source.get('date_published', 'Date unknown')
            date_accessed = source.get('date_accessed', '')
            
            # Format citation
            citation = f"{i}. **{title}**"
            if author and author != source.get('domain', ''):
                citation += f" - {author}"
            citation += f" ({date_published})"
            if url:
                citation += f" - {url}"
            if date_accessed:
                citation += f" [Accessed: {date_accessed}]"
            
            bib_text += citation + "\n\n"
        
        return bib_text
    
    # Start report generation
    print("Generating research report...")
    
    # Create bibliography
    bibliography = create_bibliography(extracted_content) if include_bibliography else []
    
    # Generate main report
    detailed_report = generate_detailed_report()
    
    # Add methodology section
    methodology_section = generate_methodology_section() if include_methodology else ""
    
    # Add bibliography section
    bibliography_section = generate_bibliography_section(bibliography)
    
    # Combine all sections
    full_report = detailed_report + methodology_section + bibliography_section
    
    # Add metadata footer
    metadata_footer = f"""
    
    ---
    
    **Report Metadata**
    - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - Research Topic: {research_topic}
    - Report Type: {report_type}
    - Target Audience: {target_audience}
    - Sources Analyzed: {len([url for url, content in extracted_content.items() if 'error' not in content])}
    - Total Sources Attempted: {len(extracted_content)}
    - Key Findings: {len(key_findings)}
    - Research Gaps Identified: {len(research_gaps)}
    """
    
    # Prepare final results
    report_results = {
        'synthesized_report': full_report,
        'detailed_analysis': detailed_report,
        'executive_summary': executive_summary,
        'source_bibliography': bibliography,
        'report_metadata': {
            'generation_timestamp': datetime.now().isoformat(),
            'report_type': report_type,
            'target_audience': target_audience,
            'include_methodology': include_methodology,
            'include_bibliography': include_bibliography,
            'sections_generated': {
                'main_report': True,
                'methodology': include_methodology,
                'bibliography': include_bibliography and len(bibliography) > 0
            },
            'statistics': {
                'total_report_words': len(full_report.split()),
                'sources_cited': len(bibliography),
                'findings_included': len(key_findings),
                'contradictions_discussed': len(contradictions),
                'consensus_points_highlighted': len(consensus_points),
                'research_gaps_identified': len(research_gaps)
            }
        }
    }
    
    print("Report generation completed!")
    print(f"  - Report words: {len(full_report.split()):,}")
    print(f"  - Sources cited: {len(bibliography)}")
    print(f"  - Sections included: Main report, {'Methodology, ' if include_methodology else ''}{'Bibliography' if include_bibliography else ''}")
    
    return json.dumps(report_results, indent=2)

def create_report_generation_tool():
    """Create LangChain StructuredTool for report generation"""
    return StructuredTool.from_function(
        name="report_generation",
        description="Generate comprehensive research reports based on analysis results",
        func=report_generation_function,
        args_schema=ReportGenerationInput,
        coroutine=None
    )