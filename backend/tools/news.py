import os
import logging
from tavily import TavilyClient

logger = logging.getLogger(__name__)

def get_latest_ai_news(topic: str = "AI tools for software engineering") -> str:
    """
    Fetch latest news about AI tools for software engineering using Tavily.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not found. Returning mock news.")
        return "Intelligence report unavailable. Satellite uplink offline."

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(topic, search_depth="advanced")
        
        results = response.get("results", [])
        if not results:
            return "No recent intelligence found."

        # Format results
        summary = "Intelligence Report:\n"
        for res in results[:3]:
            summary += f"- {res['title']}: {res['content'][:200]}...\n"
        
        return summary

    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return "Error accessing intelligence network."

