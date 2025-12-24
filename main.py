#!/usr/bin/env python3
"""
Daily Reinsurance Expert Agent
==============================

Single entrypoint that orchestrates:
1. State loading (repo memory via state.json)
2. Web research via Tavily API
3. LLM digest generation via OpenAI (structured output)
4. Email delivery via Resend or SendGrid
5. State persistence (commit back via GitHub Actions)

Table of Contents:
------------------
- Lines 30-120:   Pydantic Models (State, ResearchItem, Digest, EmailConfig)
- Lines 125-200:  Configuration & Logging Setup
- Lines 205-280:  State Management (load, validate, save)
- Lines 285-420:  Research Module (Tavily search, compression, dedupe)
- Lines 425-580:  LLM Digest Generation (OpenAI structured output)
- Lines 585-700:  Email Sending (Resend / SendGrid)
- Lines 705-800:  Main Orchestrator
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class SeenItem(BaseModel):
    """A previously seen news item (for deduplication)."""
    source: str
    url: str
    title: str
    first_seen_utc: str


class State(BaseModel):
    """
    Persistent state stored in state.json.
    
    This is the "repo memory" that tracks:
    - User's learning level (1-10)
    - Feedback to adjust complexity/tone
    - Topics already covered (avoid repetition)
    - News items already shown (deduplication)
    """
    schema_version: int = 1
    level: int = Field(default=1, ge=1, le=10)
    user_feedback: str = ""
    covered_topics: list[str] = Field(default_factory=list)
    seen_items: dict[str, SeenItem] = Field(default_factory=dict)
    last_run_utc: Optional[str] = None

    @field_validator("level", mode="before")
    @classmethod
    def clamp_level(cls, v: Any) -> int:
        """Ensure level stays within 1-10 bounds."""
        if isinstance(v, int):
            return max(1, min(10, v))
        return 1


class ResearchItem(BaseModel):
    """A single research item from Tavily search."""
    item_id: str
    source: str
    url: str
    title: str
    summary: str  # Compressed summary (max ~200 chars)
    published_date: Optional[str] = None


class MarketNewsItem(BaseModel):
    """A market news item in the digest."""
    headline: str
    summary: str
    renewals_angle: Optional[str] = None  # How it relates to 1.1/1.4/1.7 renewals


class TechnicalConcept(BaseModel):
    """Technical concept section of the digest."""
    topic: str
    explanation: str
    key_terms: list[str] = Field(default_factory=list)
    example: Optional[str] = None


class DataSaaSSection(BaseModel):
    """Data/SaaS requirements section."""
    topic: str
    explanation: str
    json_schema_example: str  # Example API schema / JSON payload
    entity_definitions: list[str] = Field(default_factory=list)
    data_cleaning_tips: list[str] = Field(default_factory=list)


class AIUseCase(BaseModel):
    """AI use case in reinsurance."""
    title: str
    description: str
    practical_application: str


class Digest(BaseModel):
    """
    The complete daily digest structure.
    
    This is the structured output we request from OpenAI.
    All sections are required for each digest.
    """
    date: str
    level: int
    greeting: str
    
    # Section 1: Market News (with Renewals focus)
    market_news: list[MarketNewsItem] = Field(min_length=1, max_length=4)
    
    # Section 2: Technical Concept (scaffolded by level)
    technical_concept: TechnicalConcept
    
    # Section 3: Data/SaaS Requirements
    data_saas: DataSaaSSection
    
    # Section 4: AI Use Cases
    ai_use_case: AIUseCase
    
    # Closing
    closing_thought: str
    new_topic_covered: str  # Topic ID to add to covered_topics


class EmailConfig(BaseModel):
    """Email configuration from environment variables."""
    provider: str = "resend"
    from_address: str
    to_address: str
    resend_api_key: Optional[str] = None
    sendgrid_api_key: Optional[str] = None


# =============================================================================
# CONFIGURATION & LOGGING
# =============================================================================

# Load environment variables from .env file (for local development)
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
STATE_FILE = Path(__file__).parent / "state.json"

# API Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "")

# Research constraints (for token cost control)
MAX_ARTICLES_PER_SOURCE = 3
MAX_TOTAL_ARTICLES = 6
MAX_SUMMARY_CHARS = 200
MAX_COVERED_TOPICS_IN_PROMPT = 20

# Mistral model selection (cost-optimized)
MISTRAL_MODEL = "mistral-small-latest"


def get_email_config() -> EmailConfig:
    """Load email configuration from environment variables."""
    provider = os.getenv("EMAIL_PROVIDER", "resend").lower()
    
    return EmailConfig(
        provider=provider,
        from_address=os.getenv("EMAIL_FROM", ""),
        to_address=os.getenv("EMAIL_TO", ""),
        resend_api_key=os.getenv("RESEND_API_KEY"),
        sendgrid_api_key=os.getenv("SENDGRID_API_KEY"),
    )


# =============================================================================
# STATE MANAGEMENT
# =============================================================================


def load_state() -> State:
    """
    Load and validate state from state.json.
    
    Returns a default state if file doesn't exist or is invalid.
    This is "Graceful Degradation" - we don't crash on bad state,
    we just start fresh.
    """
    if not STATE_FILE.exists():
        logger.warning("state.json not found, creating default state")
        return State()
    
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        state = State.model_validate(data)
        logger.info(f"Loaded state: level={state.level}, topics_covered={len(state.covered_topics)}, seen_items={len(state.seen_items)}")
        return state
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in state.json: {e}")
        return State()
    except Exception as e:
        logger.error(f"Error loading state: {e}")
        return State()


def save_state(state: State) -> None:
    """
    Atomically save state to state.json.
    
    Uses write-to-temp-then-rename pattern to prevent corruption
    if the process is interrupted mid-write.
    """
    state.last_run_utc = datetime.now(timezone.utc).isoformat()
    
    temp_file = STATE_FILE.with_suffix(".json.tmp")
    
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(state.model_dump(), f, indent=2)
        
        temp_file.replace(STATE_FILE)
        logger.info(f"State saved: level={state.level}, topics={len(state.covered_topics)}, seen={len(state.seen_items)}")
    
    except Exception as e:
        logger.error(f"Failed to save state: {e}")
        if temp_file.exists():
            temp_file.unlink()
        raise


def generate_item_id(url: str, title: str, source: str) -> str:
    """
    Generate a stable ID for deduplication.
    
    Prefers URL-based ID (most stable), falls back to title hash.
    """
    # Normalize URL
    normalized_url = url.lower().strip().rstrip("/")
    
    # Create hash from URL (primary) or title+source (fallback)
    content = normalized_url if normalized_url else f"{title.lower()}|{source.lower()}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# RESEARCH MODULE (TAVILY)
# =============================================================================


def search_tavily(query: str, site_filter: Optional[str] = None) -> list[dict]:
    """
    Search using Tavily API.
    
    Why Tavily? It's designed specifically for AI agents - returns
    clean, summarized content optimized for LLM consumption.
    
    Args:
        query: Search query
        site_filter: Optional site to restrict search to (e.g., "artemis.bm")
    
    Returns:
        List of search results with title, url, content
    """
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set, skipping research")
        return []
    
    # Build search query with site filter
    search_query = f"site:{site_filter} {query}" if site_filter else query
    
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": search_query,
                "search_depth": "basic",  # Faster, cheaper
                "max_results": MAX_ARTICLES_PER_SOURCE,
                "include_answer": False,
                "include_raw_content": False,
            },
            timeout=30,
        )
        response.raise_for_status()
        
        results = response.json().get("results", [])
        logger.info(f"Tavily search '{search_query[:50]}...' returned {len(results)} results")
        return results
    
    except requests.RequestException as e:
        logger.error(f"Tavily search failed: {e}")
        return []


def compress_summary(content: str, max_chars: int = MAX_SUMMARY_CHARS) -> str:
    """
    Compress content to fit token budget.
    
    Simple truncation with ellipsis - we let the LLM do the
    real summarization work.
    """
    if not content:
        return ""
    
    content = content.strip()
    if len(content) <= max_chars:
        return content
    
    # Truncate at word boundary
    truncated = content[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.7:  # Only if we don't lose too much
        truncated = truncated[:last_space]
    
    return truncated.rstrip(".,;:") + "..."


def fetch_research(state: State) -> list[ResearchItem]:
    """
    Fetch and filter research from target sources.
    
    Sources:
    - reinsurancene.ws: Industry news aggregator
    - artemis.bm: ILS and cat bond news
    
    Process:
    1. Search each source
    2. Filter out already-seen items
    3. Compress summaries for token efficiency
    4. Return deduplicated list
    """
    logger.info("Starting research fetch...")
    
    sources = [
        ("reinsurancene.ws", "reinsurance specialty lines marine energy news"),
        ("artemis.bm", "reinsurance catastrophe bonds ILS news"),
    ]
    
    all_items: list[ResearchItem] = []
    
    for site, query in sources:
        results = search_tavily(query, site_filter=site)
        
        for result in results:
            url = result.get("url", "")
            title = result.get("title", "")
            content = result.get("content", "")
            
            if not url or not title:
                continue
            
            # Generate stable ID for deduplication
            item_id = generate_item_id(url, title, site)
            
            # Check if already seen
            if item_id in state.seen_items:
                logger.debug(f"Skipping already-seen item: {title[:50]}...")
                continue
            
            # Compress summary
            summary = compress_summary(content)
            
            item = ResearchItem(
                item_id=item_id,
                source=site,
                url=url,
                title=title,
                summary=summary,
            )
            all_items.append(item)
            
            logger.info(f"New item: [{site}] {title[:60]}...")
    
    # Limit total items
    if len(all_items) > MAX_TOTAL_ARTICLES:
        logger.info(f"Limiting research to {MAX_TOTAL_ARTICLES} items (had {len(all_items)})")
        all_items = all_items[:MAX_TOTAL_ARTICLES]
    
    logger.info(f"Research complete: {len(all_items)} new items to process")
    return all_items


# =============================================================================
# LLM DIGEST GENERATION
# =============================================================================


def build_system_prompt(state: State) -> str:
    """
    Build the system prompt for the digest generation.
    
    Strategy: "Constrained Expert" role with clear boundaries
    - Focus ONLY on Specialty Lines (Marine, Energy, Terror, P&C)
    - Explicitly exclude Life/Health
    - Adapt complexity based on level and feedback
    """
    level_guidance = get_level_guidance(state.level)
    
    feedback_note = ""
    if state.user_feedback:
        feedback_note = f"""
USER FEEDBACK (incorporate this):
{state.user_feedback}
"""
    
    return f"""You are the Daily Reinsurance Expert Agent, a knowledgeable mentor helping a product manager learn the reinsurance industry.

SCOPE CONSTRAINTS (CRITICAL):
- Focus ONLY on Specialty Lines: Marine, Energy, Terror, Property & Casualty (P&C)
- NEVER include Life or Health insurance content
- Always relate Market News to the reinsurance renewal cycles (1.1 = Jan 1, 1.4 = April 1, 1.7 = July 1)

USER LEVEL: {state.level}/10
{level_guidance}
{feedback_note}
CONTENT REQUIREMENTS:
1. Market News: Include 2-4 items with a Renewals angle (how it affects 1.1/1.4/1.7 renewals)
2. Technical Concept: Teach ONE new concept appropriate to user's level. Avoid repeating covered topics.
3. Data/SaaS Requirements: Include realistic JSON schema examples, entity definitions, and data cleaning tips
4. AI Use Cases: One practical AI/ML application in reinsurance

STYLE:
- Mobile-friendly: short paragraphs, clear structure
- Progressive complexity based on level
- Include concrete examples, not just theory
- Use industry terminology appropriate to level

PREVIOUSLY COVERED TOPICS (do not repeat):
{', '.join(state.covered_topics[-MAX_COVERED_TOPICS_IN_PROMPT:]) if state.covered_topics else 'None yet'}
"""


def get_level_guidance(level: int) -> str:
    """Return level-appropriate guidance for content complexity."""
    if level <= 3:
        return """LEVEL GUIDANCE (Beginner):
- Explain all industry terms when first used
- Use analogies to everyday concepts
- Keep technical concepts simple and foundational
- Many examples with step-by-step explanations"""
    
    elif level <= 6:
        return """LEVEL GUIDANCE (Intermediate):
- Use industry terminology freely, explain only advanced terms
- Include more quantitative concepts
- Discuss market dynamics and business implications
- Examples can be more complex, assume basic understanding"""
    
    elif level <= 9:
        return """LEVEL GUIDANCE (Advanced):
- Full industry jargon expected
- Include complex structures (sidecars, ILWs, cat bonds)
- Discuss regulatory and capital considerations
- Quantitative examples with formulas where relevant"""
    
    else:
        return """LEVEL GUIDANCE (Expert):
- Cutting-edge topics and market innovations
- Research-level depth on technical concepts
- Strategic and analytical perspectives
- Assume comprehensive industry knowledge"""


def get_digest_json_schema() -> str:
    """
    Return the JSON schema for the Digest model.
    
    Mistral uses JSON mode (not structured output like OpenAI),
    so we include the schema in the prompt for guidance.
    """
    return '''{
  "date": "YYYY-MM-DD",
  "level": 1,
  "greeting": "Welcome message string",
  "market_news": [
    {
      "headline": "News headline",
      "summary": "Brief summary",
      "renewals_angle": "How it relates to 1.1/1.4/1.7 renewals (optional)"
    }
  ],
  "technical_concept": {
    "topic": "Topic name",
    "explanation": "Detailed explanation",
    "key_terms": ["term1", "term2"],
    "example": "Practical example (optional)"
  },
  "data_saas": {
    "topic": "Data topic name",
    "explanation": "Explanation of data requirements",
    "json_schema_example": "{ example JSON schema as string }",
    "entity_definitions": ["Entity 1: definition", "Entity 2: definition"],
    "data_cleaning_tips": ["Tip 1", "Tip 2"]
  },
  "ai_use_case": {
    "title": "AI use case title",
    "description": "Description of the use case",
    "practical_application": "How to apply it"
  },
  "closing_thought": "Closing thought string",
  "new_topic_covered": "snake_case_topic_id"
}'''


def build_user_prompt(research_items: list[ResearchItem]) -> str:
    """Build the user prompt with research context."""
    today = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")
    
    research_context = ""
    if research_items:
        research_lines = []
        for item in research_items:
            research_lines.append(f"- [{item.source}] {item.title}")
            if item.summary:
                research_lines.append(f"  Summary: {item.summary}")
        research_context = "\n".join(research_lines)
    else:
        research_context = "(No new research available today - generate digest from your knowledge)"
    
    json_schema = get_digest_json_schema()
    
    return f"""Generate today's Daily Reinsurance Digest.

DATE: {today}

TODAY'S RESEARCH ITEMS:
{research_context}

REQUIRED OUTPUT FORMAT (JSON):
{json_schema}

INSTRUCTIONS:
- Return ONLY valid JSON matching the schema above
- Include 2-4 market_news items
- new_topic_covered should be a short snake_case identifier (e.g., "treaty_quota_share")
- json_schema_example in data_saas should be a realistic API schema as a string"""


def generate_digest(state: State, research_items: list[ResearchItem]) -> Optional[Digest]:
    """
    Generate the digest using Mistral with JSON mode.
    
    Why Mistral?
    - Cost-effective alternative to OpenAI
    - Good quality outputs for structured tasks
    - JSON mode ensures valid JSON output
    
    Note: Unlike OpenAI's structured output, Mistral's JSON mode
    doesn't guarantee schema compliance, so we validate with Pydantic.
    """
    if not MISTRAL_API_KEY:
        logger.error("MISTRAL_API_KEY not set, cannot generate digest")
        return None
    
    from mistralai import Mistral
    
    client = Mistral(api_key=MISTRAL_API_KEY)
    
    system_prompt = build_system_prompt(state)
    user_prompt = build_user_prompt(research_items)
    
    # Log prompt sizes for cost awareness
    total_chars = len(system_prompt) + len(user_prompt)
    estimated_tokens = total_chars // 4  # Rough estimate
    logger.info(f"Prompt size: ~{estimated_tokens} tokens (system={len(system_prompt)}, user={len(user_prompt)} chars)")
    
    try:
        # Use JSON mode with Mistral
        response = client.chat.complete(
            model=MISTRAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        
        # Extract and parse the response
        content = response.choices[0].message.content
        
        if not content:
            logger.error("Mistral returned empty response")
            return None
        
        # Parse JSON and validate with Pydantic
        try:
            data = json.loads(content)
            digest = Digest.model_validate(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Mistral response as JSON: {e}")
            logger.debug(f"Raw response: {content[:500]}...")
            return None
        except Exception as e:
            logger.error(f"Failed to validate digest schema: {e}")
            return None
        
        # Log usage for cost tracking
        usage = response.usage
        if usage:
            logger.info(f"Mistral usage: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total tokens")
            # Estimate cost (mistral-small-latest pricing ~$0.1/1M input, $0.3/1M output)
            cost = (usage.prompt_tokens * 0.0001 + usage.completion_tokens * 0.0003) / 1000
            logger.info(f"Estimated cost: ${cost:.6f}")
        
        # Override date and level to ensure consistency
        digest.date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        digest.level = state.level
        
        logger.info(f"Digest generated successfully, new topic: {digest.new_topic_covered}")
        return digest
    
    except Exception as e:
        logger.error(f"Mistral API call failed: {e}")
        return None


# =============================================================================
# EMAIL SENDING
# =============================================================================


def format_digest_email(digest: Digest, state: State) -> tuple[str, str]:
    """
    Format the digest as a plain-text email (mobile-friendly).
    
    Returns: (subject, body)
    """
    subject = f"🎓 Daily Reinsurance Digest | {digest.date} | Level {digest.level}"
    
    # Build self-assessment link
    if GITHUB_REPOSITORY:
        edit_link = f"https://github.com/{GITHUB_REPOSITORY}/edit/main/state.json"
    else:
        edit_link = "(Set GITHUB_REPOSITORY env var for self-assessment link)"
    
    # Format market news
    market_news_text = ""
    for i, item in enumerate(digest.market_news, 1):
        market_news_text += f"\n{i}. {item.headline}\n"
        market_news_text += f"   {item.summary}\n"
        if item.renewals_angle:
            market_news_text += f"   📅 Renewals: {item.renewals_angle}\n"
    
    # Format key terms
    key_terms_text = ", ".join(digest.technical_concept.key_terms) if digest.technical_concept.key_terms else "N/A"
    
    # Format entity definitions
    entities_text = "\n".join(f"   • {e}" for e in digest.data_saas.entity_definitions) if digest.data_saas.entity_definitions else "   (none)"
    
    # Format data cleaning tips
    cleaning_text = "\n".join(f"   • {t}" for t in digest.data_saas.data_cleaning_tips) if digest.data_saas.data_cleaning_tips else "   (none)"
    
    body = f"""{digest.greeting}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📰 MARKET NEWS (Specialty Lines Focus)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{market_news_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 TECHNICAL CONCEPT: {digest.technical_concept.topic}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{digest.technical_concept.explanation}

Key Terms: {key_terms_text}

{f"Example: {digest.technical_concept.example}" if digest.technical_concept.example else ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💾 DATA & SAAS: {digest.data_saas.topic}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{digest.data_saas.explanation}

📋 Example JSON Schema:
{digest.data_saas.json_schema_example}

📖 Entity Definitions:
{entities_text}

🧹 Data Cleaning Tips:
{cleaning_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 AI USE CASE: {digest.ai_use_case.title}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{digest.ai_use_case.description}

Practical Application:
{digest.ai_use_case.practical_application}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💭 CLOSING THOUGHT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{digest.closing_thought}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 SELF-ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Current Level: {state.level}/10

Ready to level up? Edit your state.json to:
• Increment "level" (max 10)
• Add feedback in "user_feedback" (e.g., "more technical" or "more examples")

Edit here: {edit_link}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Happy learning! 🚀
"""
    
    return subject, body


def send_email_resend(to: str, from_addr: str, subject: str, body: str, api_key: str) -> bool:
    """
    Send email using Resend API.
    
    Why Resend? Modern API, generous free tier (100 emails/day),
    excellent deliverability, simple integration.
    """
    try:
        import resend
        resend.api_key = api_key
        
        response = resend.Emails.send({
            "from": from_addr,
            "to": [to],
            "subject": subject,
            "text": body,
        })
        
        logger.info(f"Email sent via Resend: {response}")
        return True
    
    except Exception as e:
        logger.error(f"Resend email failed: {e}")
        return False


def send_email_sendgrid(to: str, from_addr: str, subject: str, body: str, api_key: str) -> bool:
    """
    Send email using SendGrid API.
    
    Why SendGrid? Industry standard, generous free tier (100 emails/day),
    robust infrastructure.
    """
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        
        message = Mail(
            from_email=from_addr,
            to_emails=to,
            subject=subject,
            plain_text_content=body,
        )
        
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        
        logger.info(f"Email sent via SendGrid: status={response.status_code}")
        return response.status_code in (200, 201, 202)
    
    except Exception as e:
        logger.error(f"SendGrid email failed: {e}")
        return False


def send_digest_email(digest: Digest, state: State) -> bool:
    """
    Send the digest email using configured provider.
    
    Provider is selected via EMAIL_PROVIDER env var:
    - "resend" (default): Uses Resend API
    - "sendgrid": Uses SendGrid API
    """
    config = get_email_config()
    
    if not config.from_address or not config.to_address:
        logger.error("EMAIL_FROM and EMAIL_TO must be set")
        return False
    
    subject, body = format_digest_email(digest, state)
    
    logger.info(f"Sending email via {config.provider}: {config.from_address} -> {config.to_address}")
    
    if config.provider == "sendgrid":
        if not config.sendgrid_api_key:
            logger.error("SENDGRID_API_KEY not set")
            return False
        return send_email_sendgrid(
            config.to_address,
            config.from_address,
            subject,
            body,
            config.sendgrid_api_key,
        )
    else:
        # Default to Resend
        if not config.resend_api_key:
            logger.error("RESEND_API_KEY not set")
            return False
        return send_email_resend(
            config.to_address,
            config.from_address,
            subject,
            body,
            config.resend_api_key,
        )


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================


def update_state_with_digest(state: State, digest: Digest, research_items: list[ResearchItem]) -> None:
    """
    Update state with information from the completed digest.
    
    - Add new topic to covered_topics
    - Add research items to seen_items
    """
    # Add new topic
    if digest.new_topic_covered and digest.new_topic_covered not in state.covered_topics:
        state.covered_topics.append(digest.new_topic_covered)
        logger.info(f"Added topic: {digest.new_topic_covered}")
    
    # Add seen items
    now_utc = datetime.now(timezone.utc).isoformat()
    for item in research_items:
        if item.item_id not in state.seen_items:
            state.seen_items[item.item_id] = SeenItem(
                source=item.source,
                url=item.url,
                title=item.title,
                first_seen_utc=now_utc,
            )


def main() -> int:
    """
    Main orchestrator - runs the complete daily digest pipeline.
    
    Returns:
        0 on success, 1 on failure
    """
    logger.info("=" * 60)
    logger.info("Daily Reinsurance Expert Agent - Starting")
    logger.info("=" * 60)
    
    # Step 1: Load state
    logger.info("Step 1: Loading state...")
    state = load_state()
    
    # Step 2: Fetch research
    logger.info("Step 2: Fetching research...")
    try:
        research_items = fetch_research(state)
    except Exception as e:
        logger.warning(f"Research fetch failed (continuing without): {e}")
        research_items = []
    
    # Step 3: Generate digest
    logger.info("Step 3: Generating digest...")
    digest = generate_digest(state, research_items)
    
    if digest is None:
        logger.error("Failed to generate digest, aborting")
        return 1
    
    # Step 4: Send email
    logger.info("Step 4: Sending email...")
    email_sent = send_digest_email(digest, state)
    
    if not email_sent:
        logger.warning("Email send failed (continuing to save state)")
    
    # Step 5: Update and save state
    logger.info("Step 5: Updating state...")
    update_state_with_digest(state, digest, research_items)
    save_state(state)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Daily Digest Complete!")
    logger.info(f"  Level: {state.level}")
    logger.info(f"  Topics covered: {len(state.covered_topics)}")
    logger.info(f"  Items seen: {len(state.seen_items)}")
    logger.info(f"  Email sent: {email_sent}")
    logger.info("=" * 60)
    
    return 0 if email_sent else 1


if __name__ == "__main__":
    sys.exit(main())

