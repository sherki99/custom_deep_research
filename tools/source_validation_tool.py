import os
import re
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
import json

load_dotenv(override=True)

class SourceValidationInput(BaseModel):
    """Input schema for Source Validation Tool"""
    search_results: List[Dict[str, Any]] = Field(..., description="List of search results to validate")
    min_credibility_threshold: float = Field(default=0.3, description="Minimum credibility score to accept sources")
    llm_weight: float = Field(default=0.4, description="Weight for LLM credibility assessment (0.0-1.0)")
    check_accessibility: bool = Field(default=True, description="Whether to check URL accessibility")
    timeout: int = Field(default=10, description="Timeout for URL accessibility check")

def source_validation_function(
    search_results: List[Dict[str, Any]],
    min_credibility_threshold: float = 0.3,
    llm_weight: float = 0.4,
    check_accessibility: bool = True,
    timeout: int = 10
) -> str:
    """
    Validate search results and assign credibility scores using domain analysis and LLM assessment.
    """
    
    # Initialize LLM
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_deployment=os.getenv("LLM_DEPLOYMENT_NAME")
        )
    except Exception as e:
        return json.dumps({"error": f"LLM initialization failed: {str(e)}", "results": {}})
    
    # Known credibility domains
    high_credibility_domains = {
        'nature.com', 'science.org', 'cell.com', 'nejm.org', 'bmj.com',
        'who.int', 'cdc.gov', 'nih.gov', 'fda.gov', 'europa.eu',
        'reuters.com', 'apnews.com', 'bbc.com', 'economist.com',
        'ft.com', 'wsj.com', 'nytimes.com', 'theguardian.com',
        'wikipedia.org', 'britannica.com', 'arxiv.org', 'pubmed.ncbi.nlm.nih.gov'
    }
    
    low_credibility_domains = {
        'dailymail.co.uk', 'breitbart.com', 'infowars.com',
        'naturalnews.com', 'mercola.com', 'zerohedge.com', "youtube.com"
    }
    
    academic_patterns = [r'\.edu$', r'\.ac\.', r'\.org$']
    government_patterns = [r'\.gov$', r'\.gov\.']
    
    def extract_domain(url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower().replace('www.', '')
        except:
            return ""
    
    def assess_domain_credibility(domain: str) -> float:
        """Assess credibility based on domain"""
        if domain in high_credibility_domains:
            return 0.9
        elif domain in low_credibility_domains:
            return 0.2
        
        # Check for academic/institutional patterns
        for pattern in academic_patterns:
            if re.search(pattern, domain):
                return 0.8
        
        for pattern in government_patterns:
            if re.search(pattern, domain):
                return 0.85
        
        return 0.5  # Default score
    
    def check_url_accessibility(url: str) -> Dict[str, Any]:
        """Check if URL is accessible"""
        if not check_accessibility:
            return {'accessible': True, 'status_code': 200}
            
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            return {
                'accessible': response.status_code == 200,
                'status_code': response.status_code,
                'final_url': response.url,
                'content_type': response.headers.get('content-type', '')
            }
        except requests.RequestException as e:
            return {
                'accessible': False,
                'error': str(e),
                'status_code': None
            }
    
    def assess_content_quality(result: Dict[str, Any]) -> float:
        """Assess content quality based on metadata"""
        score = 0.0
        
        # Title quality
        title = result.get('title', '')
        if title:
            title_len = len(title)
            if 20 <= title_len <= 120:
                score += 0.3
            elif 10 <= title_len < 200:
                score += 0.2
            else:
                score += 0.1
            
            # Check for clickbait
            clickbait_words = ['shocking', 'unbelievable', 'amazing', 'incredible', 
                             'you won\'t believe', 'doctors hate', 'one weird trick']
            has_clickbait = any(word.lower() in title.lower() for word in clickbait_words)
            if not has_clickbait:
                score += 0.2
        
        # Snippet quality
        snippet = result.get('snippet', '')
        if snippet and len(snippet) > 50:
            score += 0.2
        
        # Date availability
        if result.get('date'):
            score += 0.1
        
        return min(1.0, score)
    
    def use_llm_for_credibility(result: Dict[str, Any]) -> float:
        """Use LLM to assess source credibility"""
        prompt = f"""Assess the credibility of this source on a scale of 0.0 to 1.0:

Title: {result.get('title', 'No title')}
Snippet: {result.get('snippet', 'No snippet')}
Source: {result.get('source', 'Unknown source')}
URL: {result.get('url', 'No URL')}

Consider:
- Authority and reputation of the source
- Quality of the title (avoid clickbait)
- Informativeness of the snippet
- Professional presentation

Return only a number between 0.0 and 1.0:
- 0.9-1.0: Highly credible (academic, official, reputable news)
- 0.7-0.8: Good credibility (established sources)
- 0.5-0.6: Moderate credibility
- 0.3-0.4: Low credibility
- 0.0-0.2: Very low credibility

Score:"""
        
        try:
            response = llm.invoke([{"role": "user", "content": prompt}])
            score_text = response.content.strip()
            
            # Extract number from response
            numbers = re.findall(r'0\.\d+|1\.0|0|1', score_text)
            if numbers:
                return min(1.0, max(0.0, float(numbers[0])))
            else:
                return 0.5
                
        except Exception as e:
            print(f"LLM credibility assessment failed: {e}")
            return 0.5
    
    # Process each search result
    validated_sources = {}
    credibility_scores = {}
    removed_sources = []
    
    for idx, result in enumerate(search_results):
        url = result.get('url', '')
        domain = extract_domain(url)
        
        if not url:
            removed_sources.append({
                'url': url,
                'reason': 'No URL provided',
                'title': result.get('title', 'Unknown')
            })
            continue
        
        # Check accessibility
        accessibility = check_url_accessibility(url)
        
        # Assess domain credibility
        domain_score = assess_domain_credibility(domain)
        
        # Assess content quality
        content_score = assess_content_quality(result)
        
        # Use LLM for credibility assessment
        llm_score = use_llm_for_credibility(result)
        
        # Calculate final score with weights
        rule_based_score = (domain_score * 0.5) + (content_score * 0.5)
        final_score = (rule_based_score * (1 - llm_weight)) + (llm_score * llm_weight)
        
        # Validate source
        if final_score >= min_credibility_threshold and accessibility.get('accessible', True):
            validated_sources[url] = {
                'title': result.get('title', ''),
                'snippet': result.get('snippet', ''),
                'source': result.get('source', ''),
                'domain': domain,
                'date': result.get('date', ''),
                'accessibility': accessibility,
                'domain_score': domain_score,
                'content_score': content_score,
                'llm_score': llm_score,
                'final_score': final_score,
                'validation_timestamp': datetime.now().isoformat()
            }
            credibility_scores[url] = final_score
        else:
            reason = []
            if final_score < min_credibility_threshold:
                reason.append(f'Low credibility score: {final_score:.2f}')
            if not accessibility.get('accessible', True):
                reason.append(f'Not accessible: {accessibility.get("error", "Unknown error")}')
            
            removed_sources.append({
                'url': url,
                'reason': '; '.join(reason),
                'title': result.get('title', 'Unknown'),
                'credibility_score': final_score
            })
    
    # Calculate statistics
    credibility_average = sum(credibility_scores.values()) / len(credibility_scores) if credibility_scores else 0.0
    
    return json.dumps({
        'validated_sources': validated_sources,
        'credibility_scores': credibility_scores,
        'removed_sources': removed_sources,
        'credibility_average': credibility_average,
        'total_validated': len(validated_sources),
        'total_removed': len(removed_sources),
        'validation_settings': {
            'min_threshold': min_credibility_threshold,
            'llm_weight': llm_weight,
            'check_accessibility': check_accessibility
        }
    }, indent=2)

def create_source_validation_tool():
    """Create LangChain StructuredTool for source validation"""
    return StructuredTool.from_function(
        name="source_validation",
        description="Validate search results and assign credibility scores using domain analysis and LLM assessment",
        func=source_validation_function,
        args_schema=SourceValidationInput,
        coroutine=None
    )