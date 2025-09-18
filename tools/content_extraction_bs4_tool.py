import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, urljoin
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
import json
import re
import time

load_dotenv(override=True)

class ContentExtractionInput(BaseModel):
    """Input schema for Content Extraction Tool"""
    urls: List[str] = Field(..., description="List of URLs to extract content from")
    max_content_length: int = Field(default=20000, description="Maximum content length per URL")
    timeout: int = Field(default=15, description="Timeout for each URL request")
    extract_metadata: bool = Field(default=True, description="Whether to extract page metadata")
    extract_links: bool = Field(default=False, description="Whether to extract internal links")
    delay_between_requests: float = Field(default=1.0, description="Delay between requests in seconds")

def content_extraction_function(
    urls: List[str],
    max_content_length: int = 20000,
    timeout: int = 15,
    extract_metadata: bool = True,
    extract_links: bool = False,
    delay_between_requests: float = 1.0
) -> str:
    """
    Extract content from a list of URLs with comprehensive text processing and metadata extraction.
    """
    
    # Headers to mimic a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    def clean_text(text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common unwanted patterns
        text = re.sub(r'^\s*•\s*', '', text, flags=re.MULTILINE)  # Bullet points
        text = re.sub(r'^\s*-\s*', '', text, flags=re.MULTILINE)  # Dashes
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines
        
        return text
    
    def extract_page_content(url: str, html_content: str) -> Dict[str, Any]:
        """Extract structured content from HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Extract title
            title = ""
            if soup.title:
                title = clean_text(soup.title.string or "")
            elif soup.find('h1'):
                title = clean_text(soup.find('h1').get_text())
            
            # Extract meta description
            description = ""
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                description = clean_text(meta_desc.get('content', ''))
            
            # Extract main content
            main_content = ""
            
            # Try to find main content areas
            content_selectors = [
                'main', 'article', '[role="main"]', '.content', '#content',
                '.post-content', '.entry-content', '.article-body', '.story-body'
            ]
            
            content_element = None
            for selector in content_selectors:
                content_element = soup.select_one(selector)
                if content_element:
                    break
            
            if content_element:
                main_content = clean_text(content_element.get_text())
            else:
                # Fallback to body content
                body = soup.find('body')
                if body:
                    main_content = clean_text(body.get_text())
            
            # Extract headings
            headings = []
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                heading_text = clean_text(heading.get_text())
                if heading_text:
                    headings.append({
                        'level': heading.name,
                        'text': heading_text
                    })
            
            # Extract paragraphs (first few for summary)
            paragraphs = []
            for p in soup.find_all('p')[:5]:  # First 5 paragraphs
                p_text = clean_text(p.get_text())
                if len(p_text) > 50:  # Only substantial paragraphs
                    paragraphs.append(p_text)
            
            # Extract links if requested
            links = []
            if extract_links:
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    link_text = clean_text(link.get_text())
                    if link_text and href:
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            href = urljoin(url, href)
                        links.append({
                            'url': href,
                            'text': link_text
                        })
            
            # Calculate content statistics
            word_count = len(main_content.split()) if main_content else 0
            char_count = len(main_content) if main_content else 0
            
            # Truncate content if too long
            if main_content and len(main_content) > max_content_length:
                main_content = main_content[:max_content_length] + "... [truncated]"
            
            # Extract additional metadata
            metadata = {}
            if extract_metadata:
                # Author
                author_meta = soup.find('meta', attrs={'name': 'author'})
                if author_meta:
                    metadata['author'] = author_meta.get('content', '')
                
                # Published date
                date_selectors = [
                    'meta[name="article:published_time"]',
                    'meta[property="article:published_time"]',
                    'time[datetime]',
                    '.date', '.published'
                ]
                
                for selector in date_selectors:
                    date_element = soup.select_one(selector)
                    if date_element:
                        if date_element.get('content'):
                            metadata['published_date'] = date_element.get('content')
                        elif date_element.get('datetime'):
                            metadata['published_date'] = date_element.get('datetime')
                        elif date_element.get_text():
                            metadata['published_date'] = clean_text(date_element.get_text())
                        break
                
                # Keywords
                keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
                if keywords_meta:
                    metadata['keywords'] = keywords_meta.get('content', '')
                
                # Language
                lang = soup.find('html', attrs={'lang': True})
                if lang:
                    metadata['language'] = lang.get('lang', '')
            
            return {
                'title': title,
                'description': description,
                'content': main_content,
                'headings': headings,
                'paragraphs': paragraphs,
                'links': links,
                'metadata': metadata,
                'stats': {
                    'word_count': word_count,
                    'char_count': char_count,
                    'heading_count': len(headings),
                    'paragraph_count': len(paragraphs),
                    'link_count': len(links)
                },
                'extraction_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': f'Content parsing failed: {str(e)}',
                'title': '',
                'description': '',
                'content': '',
                'headings': [],
                'paragraphs': [],
                'links': [],
                'metadata': {},
                'stats': {},
                'extraction_timestamp': datetime.now().isoformat()
            }
    
    # Process each URL
    extracted_content = {}
    failed_extractions = []
    content_stats = {
        'total_urls': len(urls),
        'successful_extractions': 0,
        'failed_extractions': 0,
        'total_word_count': 0,
        'total_char_count': 0,
        'average_content_length': 0
    }
    
    for idx, url in enumerate(urls):
        print(f"Extracting content from {idx + 1}/{len(urls)}: {url}")
        
        try:
            # Add delay between requests
            if idx > 0:
                time.sleep(delay_between_requests)
            
            # Fetch the webpage
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'html' not in content_type:
                failed_extractions.append(url)
                print(f"  Skipped: Not HTML content ({content_type})")
                continue
            
            # Extract content
            content_data = extract_page_content(url, response.text)
            
            # Store results
            extracted_content[url] = content_data
            
            # Update stats
            if 'error' not in content_data:
                content_stats['successful_extractions'] += 1
                stats = content_data.get('stats', {})
                content_stats['total_word_count'] += stats.get('word_count', 0)
                content_stats['total_char_count'] += stats.get('char_count', 0)
                
                print(f"  Success: {stats.get('word_count', 0)} words, {len(content_data.get('title', ''))} chars title")
            else:
                content_stats['failed_extractions'] += 1
                failed_extractions.append(url)
                print(f"  Failed: {content_data.get('error', 'Unknown error')}")
                
        except requests.RequestException as e:
            failed_extractions.append(url)
            content_stats['failed_extractions'] += 1
            print(f"  Failed: Request error - {str(e)}")
            
            extracted_content[url] = {
                'error': f'Request failed: {str(e)}',
                'title': '',
                'description': '',
                'content': '',
                'headings': [],
                'paragraphs': [],
                'links': [],
                'metadata': {},
                'stats': {},
                'extraction_timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            failed_extractions.append(url)
            content_stats['failed_extractions'] += 1
            print(f"  Failed: Unexpected error - {str(e)}")
            
            extracted_content[url] = {
                'error': f'Unexpected error: {str(e)}',
                'title': '',
                'description': '',
                'content': '',
                'headings': [],
                'paragraphs': [],
                'links': [],
                'metadata': {},
                'stats': {},
                'extraction_timestamp': datetime.now().isoformat()
            }
    
    # Calculate final statistics
    if content_stats['successful_extractions'] > 0:
        content_stats['average_content_length'] = int(content_stats['total_char_count'] / content_stats['successful_extractions'])
    
    return json.dumps({
        'extracted_content': extracted_content,
        'content_stats': content_stats,
        'failed_extractions': failed_extractions,
        'extraction_settings': {
            'max_content_length': max_content_length,
            'timeout': timeout,
            'extract_metadata': extract_metadata,
            'extract_links': extract_links,
            'delay_between_requests': delay_between_requests
        }
    }, indent=2)

def create_content_extraction_tool():
    """Create LangChain StructuredTool for content extraction"""
    return StructuredTool.from_function(
        name="content_extraction",
        description="Extract content from URLs with comprehensive text processing and metadata extraction",
        func=content_extraction_function,
        args_schema=ContentExtractionInput,
        coroutine=None
    )
