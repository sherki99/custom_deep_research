import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
import re
from collections import Counter

load_dotenv(override=True)

class AnalysisSynthesisInput(BaseModel):
    """Input schema for Analysis and Synthesis Tool"""
    research_topic: str = Field(..., description="The original research topic")
    extracted_content: Dict[str, Dict[str, Any]] = Field(..., description="Content extracted from URLs")
    search_queries: List[str] = Field(default=[], description="Original search queries used")
    analysis_depth: str = Field(default="comprehensive", description="shallow, standard, comprehensive, deep")
    focus_areas: List[str] = Field(default=[], description="Specific areas to focus analysis on")
    include_contradictions: bool = Field(default=True, description="Whether to identify contradictions")
    include_gaps: bool = Field(default=True, description="Whether to identify research gaps")
    max_findings: int = Field(default=10, description="Maximum number of key findings to extract")

def analysis_synthesis_function(
    research_topic: str,
    extracted_content: Dict[str, Dict[str, Any]],
    search_queries: List[str] = None,
    analysis_depth: str = "comprehensive",
    focus_areas: List[str] = None,
    include_contradictions: bool = True,
    include_gaps: bool = True,
    max_findings: int = 10
) -> str:
    """
    Analyze and synthesize extracted content to generate insights, findings, and reports.
    """
    
    if search_queries is None:
        search_queries = []
    if focus_areas is None:
        focus_areas = []
    
    # Initialize LLM
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
        api_version=os.getenv("AZURE_API_VERSION"),
        azure_deployment=os.getenv("LLM_DEPLOYMENT_NAME"),
        temperature=0.1
    )
    
    def preprocess_content(content_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Preprocess extracted content for analysis"""
        
        processed_content = {
            'sources': [],
            'total_word_count': 0,
            'total_sources': 0,
            'successful_sources': 0,
            'content_by_source': {},
            'all_titles': [],
            'all_headings': [],
            'combined_content': ""
        }
        
        for url, content_data in content_dict.items():
            if 'error' in content_data:
                continue
                
            processed_content['successful_sources'] += 1
            
            source_info = {
                'url': url,
                'title': content_data.get('title', ''),
                'description': content_data.get('description', ''),
                'content': content_data.get('content', ''),
                'word_count': content_data.get('stats', {}).get('word_count', 0),
                'headings': content_data.get('headings', []),
                'metadata': content_data.get('metadata', {}),
                'domain': url.split('/')[2] if '//' in url else url
            }
            
            processed_content['sources'].append(source_info)
            processed_content['content_by_source'][url] = source_info
            processed_content['total_word_count'] += source_info['word_count']
            processed_content['all_titles'].append(source_info['title'])
            processed_content['all_headings'].extend([h.get('text', '') for h in source_info['headings']])
            processed_content['combined_content'] += f"\n\n--- {source_info['title']} ---\n{source_info['content']}"
        
        processed_content['total_sources'] = len(content_dict)
        
        return processed_content
    
    def extract_key_findings(processed_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key findings from the content using LLM"""
        
        findings_prompt = ChatPromptTemplate.from_template("""
        You are an expert research analyst. Analyze the following content about "{research_topic}" and extract the most important key findings.
        
        Research Topic: {research_topic}
        Focus Areas: {focus_areas}
        Analysis Depth: {analysis_depth}
        
        Content to analyze:
        {content}
        
        Please extract {max_findings} key findings. For each finding:
        1. Provide a clear, concise statement
        2. Include supporting evidence from the sources
        3. Rate the confidence level (1-5, where 5 is highest)
        4. Identify which sources support this finding
        5. Categorize the finding type (fact, trend, opinion, prediction, etc.)
        
        Return your response as a JSON array with this structure:
        [
            {{
                "finding": "Clear statement of the finding",
                "evidence": "Supporting evidence from sources",
                "confidence": 4,
                "supporting_sources": ["source1", "source2"],
                "category": "fact/trend/opinion/prediction/methodology/other",
                "importance": "high/medium/low"
            }}
        ]
        
        Only return valid JSON, no additional text.
        """)
        
        try:
            # Truncate content if too long (keep within token limits)
            content = processed_content['combined_content']
            if len(content) > 12000:  # Rough character limit
                content = content[:12000] + "... [truncated for analysis]"
            
            response = llm.invoke(
                findings_prompt.format(
                    research_topic=research_topic,
                    focus_areas=", ".join(focus_areas) if focus_areas else "General analysis",
                    analysis_depth=analysis_depth,
                    content=content,
                    max_findings=max_findings
                )
            )
            
            # Parse JSON response
            findings_text = response.content.strip()
            if findings_text.startswith('```json'):
                findings_text = findings_text[7:-3]
            elif findings_text.startswith('```'):
                findings_text = findings_text[3:-3]
            
            findings = json.loads(findings_text)
            return findings
            
        except Exception as e:
            print(f"Error extracting key findings: {str(e)}")
            return []
    
    def identify_contradictions(processed_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify contradictions or conflicting information"""
        
        if not include_contradictions or len(processed_content['sources']) < 2:
            return []
        
        contradictions_prompt = ChatPromptTemplate.from_template("""
        You are an expert research analyst. Analyze the following content about "{research_topic}" and identify any contradictions, conflicting statements, or differing viewpoints between sources.
        
        Research Topic: {research_topic}
        
        Content from multiple sources:
        {content}
        
        Please identify contradictions between sources. For each contradiction:
        1. Describe the conflicting statements
        2. Identify which sources support each side
        3. Assess the severity of the contradiction
        4. Suggest possible explanations for the contradiction
        
        Return your response as a JSON array with this structure:
        [
            {{
                "contradiction": "Description of the conflicting information",
                "source_a": {{"position": "Position A", "sources": ["source1"]}},
                "source_b": {{"position": "Position B", "sources": ["source2"]}},
                "severity": "high/medium/low",
                "possible_explanations": ["Different time periods", "Different methodologies", etc.]
            }}
        ]
        
        Only return valid JSON, no additional text. If no contradictions found, return an empty array [].
        """)
        
        try:
            content = processed_content['combined_content']
            if len(content) > 10000:
                content = content[:10000] + "... [truncated for analysis]"
            
            response = llm.invoke(
                contradictions_prompt.format(
                    research_topic=research_topic,
                    content=content
                )
            )
            
            contradictions_text = response.content.strip()
            if contradictions_text.startswith('```json'):
                contradictions_text = contradictions_text[7:-3]
            elif contradictions_text.startswith('```'):
                contradictions_text = contradictions_text[3:-3]
            
            contradictions = json.loads(contradictions_text)
            return contradictions
            
        except Exception as e:
            print(f"Error identifying contradictions: {str(e)}")
            return []
    
    def identify_consensus_points(processed_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify points where multiple sources agree"""
        
        consensus_prompt = ChatPromptTemplate.from_template("""
        You are an expert research analyst. Analyze the following content about "{research_topic}" and identify points where multiple sources agree or show consensus.
        
        Research Topic: {research_topic}
        
        Content from multiple sources:
        {content}
        
        Please identify consensus points where multiple sources agree. For each consensus point:
        1. Describe the agreed-upon point
        2. List which sources support this consensus
        3. Rate the strength of consensus (1-5)
        4. Categorize the type of consensus
        
        Return your response as a JSON array with this structure:
        [
            {{
                "consensus_point": "Description of what sources agree on",
                "supporting_sources": ["source1", "source2", "source3"],
                "strength": 4,
                "category": "fact/methodology/trend/recommendation/other",
                "evidence": "Brief summary of supporting evidence"
            }}
        ]
        
        Only return valid JSON, no additional text. Focus on finding strong consensus points supported by multiple sources.
        """)
        
        try:
            content = processed_content['combined_content']
            if len(content) > 10000:
                content = content[:10000] + "... [truncated for analysis]"
            
            response = llm.invoke(
                consensus_prompt.format(
                    research_topic=research_topic,
                    content=content
                )
            )
            
            consensus_text = response.content.strip()
            if consensus_text.startswith('```json'):
                consensus_text = consensus_text[7:-3]
            elif consensus_text.startswith('```'):
                consensus_text = consensus_text[3:-3]
            
            consensus_points = json.loads(consensus_text)
            return consensus_points
            
        except Exception as e:
            print(f"Error identifying consensus points: {str(e)}")
            return []
    
    def identify_research_gaps(processed_content: Dict[str, Any]) -> List[str]:
        """Identify gaps in the research or areas needing further investigation"""
        
        if not include_gaps:
            return []
        
        gaps_prompt = ChatPromptTemplate.from_template("""
        You are an expert research analyst. Analyze the following content about "{research_topic}" and identify research gaps, unanswered questions, or areas that need further investigation.
        
        Research Topic: {research_topic}
        Original Search Queries: {search_queries}
        
        Content analyzed:
        {content}
        
        Please identify research gaps and areas for further investigation. Consider:
        1. Questions raised but not answered
        2. Areas with insufficient information
        3. Methodological limitations mentioned
        4. Future research directions suggested
        5. Missing perspectives or stakeholder views
        
        Return your response as a JSON array of strings:
        [
            "Gap 1: Description of research gap or area needing investigation",
            "Gap 2: Another research gap identified",
            ...
        ]
        
        Only return valid JSON, no additional text. Focus on actionable research gaps.
        """)
        
        try:
            content = processed_content['combined_content']
            if len(content) > 8000:
                content = content[:8000] + "... [truncated for analysis]"
            
            response = llm.invoke(
                gaps_prompt.format(
                    research_topic=research_topic,
                    search_queries=", ".join(search_queries) if search_queries else "Not provided",
                    content=content
                )
            )
            
            gaps_text = response.content.strip()
            if gaps_text.startswith('```json'):
                gaps_text = gaps_text[7:-3]
            elif gaps_text.startswith('```'):
                gaps_text = gaps_text[3:-3]
            
            research_gaps = json.loads(gaps_text)
            return research_gaps
            
        except Exception as e:
            print(f"Error identifying research gaps: {str(e)}")
            return []
    
    def generate_executive_summary(processed_content: Dict[str, Any], findings: List[Dict]) -> str:
        """Generate an executive summary of the research"""
        
        summary_prompt = ChatPromptTemplate.from_template("""
        You are an expert research analyst. Create a concise executive summary of the research on "{research_topic}".
        
        Research Topic: {research_topic}
        Sources Analyzed: {source_count}
        Total Content: {word_count} words
        
        Key Findings:
        {key_findings}
        
        Create an executive summary that:
        1. Provides a brief overview of the research conducted
        2. Highlights the most important findings (3-5 key points)
        3. Notes any significant trends or patterns
        4. Mentions data quality and source reliability
        5. Keeps the summary concise (200-300 words)
        
        Write in a professional, clear style suitable for stakeholders who need a quick overview.
        """)
        
        try:
            findings_summary = "\n".join([
                f"- {finding['finding']} (Confidence: {finding['confidence']}/5)"
                for finding in findings[:5]  # Top 5 findings
            ])
            
            response = llm.invoke(
                summary_prompt.format(
                    research_topic=research_topic,
                    source_count=processed_content['successful_sources'],
                    word_count=processed_content['total_word_count'],
                    key_findings=findings_summary
                )
            )
            
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating executive summary: {str(e)}")
            return f"Executive summary generation failed: {str(e)}"
    
    def calculate_quality_metrics(processed_content: Dict[str, Any], findings: List[Dict]) -> Dict[str, float]:
        """Calculate research quality metrics"""
        
        metrics = {
            'source_diversity_score': 0.0,
            'information_depth_score': 0.0,
            'credibility_average': 0.0,
            'coverage_completeness': 0.0
        }
        
        try:
            # Source diversity (based on domains)
            domains = set()
            for source in processed_content['sources']:
                domains.add(source['domain'])
            
            if processed_content['successful_sources'] > 0:
                metrics['source_diversity_score'] = min(len(domains) / processed_content['successful_sources'], 1.0)
            
            # Information depth (based on content length and findings)
            avg_content_length = processed_content['total_word_count'] / max(processed_content['successful_sources'], 1)
            depth_score = min(avg_content_length / 1000, 1.0)  # Normalize to 1000 words
            findings_score = min(len(findings) / 10, 1.0)  # Normalize to 10 findings
            metrics['information_depth_score'] = (depth_score + findings_score) / 2
            
            # Credibility (based on high-confidence findings)
            if findings:
                high_confidence_findings = [f for f in findings if f.get('confidence', 0) >= 4]
                metrics['credibility_average'] = len(high_confidence_findings) / len(findings)
            
            # Coverage completeness (based on successful extractions)
            if processed_content['total_sources'] > 0:
                metrics['coverage_completeness'] = processed_content['successful_sources'] / processed_content['total_sources']
            
        except Exception as e:
            print(f"Error calculating quality metrics: {str(e)}")
        
        return metrics
    
    # Start analysis
    print("Starting content analysis and synthesis...")
    
    # Preprocess content
    processed_content = preprocess_content(extracted_content)
    
    if processed_content['successful_sources'] == 0:
        return json.dumps({
            'error': 'No successful content extractions to analyze',
            'key_findings': [],
            'contradictions': [],
            'consensus_points': [],
            'research_gaps': [],
            'executive_summary': '',
            'quality_metrics': {},
            'analysis_stats': {
                'sources_analyzed': 0,
                'total_word_count': 0,
                'findings_extracted': 0,
                'contradictions_found': 0,
                'consensus_points_found': 0,
                'research_gaps_identified': 0
            }
        }, indent=2)
    
    print(f"Analyzing {processed_content['successful_sources']} sources with {processed_content['total_word_count']:,} words...")
    
    # Extract key findings
    print("Extracting key findings...")
    key_findings = extract_key_findings(processed_content)
    
    # Identify contradictions
    print("Identifying contradictions...")
    contradictions = identify_contradictions(processed_content)
    
    # Identify consensus points
    print("Identifying consensus points...")
    consensus_points = identify_consensus_points(processed_content)
    
    # Identify research gaps
    print("Identifying research gaps...")
    research_gaps = identify_research_gaps(processed_content)
    
    # Generate executive summary
    print("Generating executive summary...")
    executive_summary = generate_executive_summary(processed_content, key_findings)
    
    # Calculate quality metrics
    print("Calculating quality metrics...")
    quality_metrics = calculate_quality_metrics(processed_content, key_findings)
    
    # Compile results
    analysis_results = {
        'key_findings': key_findings,
        'contradictions': contradictions,
        'consensus_points': consensus_points,
        'research_gaps': research_gaps,
        'executive_summary': executive_summary,
        'quality_metrics': quality_metrics,
        'analysis_stats': {
            'sources_analyzed': processed_content['successful_sources'],
            'total_sources_attempted': processed_content['total_sources'],
            'total_word_count': processed_content['total_word_count'],
            'findings_extracted': len(key_findings),
            'contradictions_found': len(contradictions),
            'consensus_points_found': len(consensus_points),
            'research_gaps_identified': len(research_gaps),
            'analysis_timestamp': datetime.now().isoformat()
        },
        'source_summary': [
            {
                'url': source['url'],
                'title': source['title'],
                'domain': source['domain'],
                'word_count': source['word_count']
            }
            for source in processed_content['sources']
        ]
    }
    
    print("Analysis and synthesis completed!")
    print(f"  - Key findings: {len(key_findings)}")
    print(f"  - Contradictions: {len(contradictions)}")
    print(f"  - Consensus points: {len(consensus_points)}")
    print(f"  - Research gaps: {len(research_gaps)}")
    print(f"  - Source diversity: {quality_metrics['source_diversity_score']:.2f}")
    print(f"  - Information depth: {quality_metrics['information_depth_score']:.2f}")
    
    return json.dumps(analysis_results, indent=2)

def create_analysis_synthesis_tool():
    """Create LangChain StructuredTool for content analysis and synthesis"""
    return StructuredTool.from_function(
        name="analysis_synthesis",
        description="Analyze and synthesize extracted content to generate insights, findings, and reports",
        func=analysis_synthesis_function,
        args_schema=AnalysisSynthesisInput,
        coroutine=None
    )