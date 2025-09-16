from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class OnlineResearchState(BaseModel):
    """
    Online research state for managing the entire research pipeline.
    """

    # --- Input parameters ---
    research_topic: str
    search_depth: str = Field(default="medium", description="shallow, medium, deep")
    max_sources_per_query: int = Field(default=10)
    max_total_sources: int = Field(default=50)
    language: str = Field(default="en")
    geographic_focus: Optional[str] = Field(default=None, description="IT, US, global")
    date_filter: Optional[str] = Field(default="all", description="day, week, month, year, all")
    source_types: List[str] = Field(default=["web", "news"], description="web, news, scholar")

    # --- Query generation ---
    search_queries: List[str] = Field(default=[], description="Generated queries for the topic")
    query_strategy: str = Field(default="comprehensive", description="focused, comprehensive, exploratory")

    # --- Search results ---
    raw_search_results: List[Dict[str, Any]] = Field(default=[], description="Raw results from search tool")
    filtered_results: List[Dict[str, Any]] = Field(default=[], description="Results after filtering")
    selected_urls: List[str] = Field(default=[], description="URLs selected for further processing")

    # --- Content extraction ---
    extracted_content: Dict[str, Dict[str, Any]] = Field(default={}, description="Content extracted from URLs")
    content_stats: Dict[str, int] = Field(default={}, description="Stats about extracted content")
    failed_extractions: List[str] = Field(default=[], description="URLs failed to extract content")

    # --- Source validation ---
    validated_sources: Dict[str, Dict[str, Any]] = Field(default={}, description="Validated source metadata")
    source_credibility_scores: Dict[str, float] = Field(default={}, description="Credibility scores per source")
    removed_sources: List[Dict[str, str]] = Field(default=[], description="Sources removed after validation")

    # --- Analysis and synthesis ---
    key_findings: List[Dict[str, Any]] = Field(default=[], description="Key findings from analysis")
    contradictions: List[Dict[str, Any]] = Field(default=[], description="Contradictory points found")
    consensus_points: List[Dict[str, Any]] = Field(default=[], description="Points most sources agree on")
    research_gaps: List[str] = Field(default=[], description="Gaps identified in current research")

    # --- Final output ---
    synthesized_report: str = Field(default="", description="Final synthesized report")
    executive_summary: str = Field(default="", description="Short executive summary")
    detailed_analysis: str = Field(default="", description="Detailed analysis report")
    source_bibliography: List[Dict[str, str]] = Field(default=[], description="List of sources and references")

    # --- Processing metadata ---
    current_step: str = Field(default="starting", description="Current step in research pipeline")
    processing_time: Dict[str, float] = Field(default={}, description="Time spent on each step")
    errors: List[str] = Field(default=[], description="Errors encountered")
    warnings: List[str] = Field(default=[], description="Warnings encountered")
    total_processing_time: float = Field(default=0.0, description="Total processing time")

    # --- Research quality metrics ---
    source_diversity_score: float = Field(default=0.0, description="Diversity of sources used")
    information_depth_score: float = Field(default=0.0, description="Depth of information gathered")
    credibility_average: float = Field(default=0.0, description="Average credibility of sources")
    coverage_completeness: float = Field(default=0.0, description="Completeness of topic coverage")





# 🔎 Extra Search Tools

# NewsSearchTool → better news coverage (GNews, Serper news).

# YouTubeSearchTool → find video content.

# PatentSearchTool → get patents (Google Patents, Lens.org).

# 📄 Extra Content Extraction Tools

# PDFExtractorTool → parse academic papers or reports (PyPDF2, Unstructured.io).

# YouTubeTranscriptTool → get transcripts from YouTube.

# ImageTextExtractorTool → OCR for extracting text from images (Tesseract, AWS Rekognition).

# 🛡️ Extra Validation Tools

# FactCheckTool → check claims with fact-check APIs.

# DomainCredibilityTool → rank websites for trustworthiness.

# BiasDetectionTool → flag potential political/commercial bias.

# 📊 Extra Analysis Tools

# ContradictionDetectorTool → detect conflicting claims across sources.

# ConsensusFinderTool → highlight what most sources agree on.

# GapIdentifierTool → find missing or underreported aspects.

# TrendAnalysisTool → show time-based changes (e.g., last week vs last year).

# 📑 Extra Output Tools

# BibliographyFormatterTool → format references in APA, MLA, etc.

# VisualizationTool → generate charts/graphs for stats (diversity, credibility).

# SlideDeckGeneratorTool → export results as a PowerPoint/Keynote-style summary.

# ExecutiveBriefTool → create a very short 1-page briefing for quick reading.