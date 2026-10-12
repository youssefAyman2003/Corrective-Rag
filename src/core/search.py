import logging
import os
from typing import Any, List, Dict
from config.settings import get_settings

logger = logging.getLogger(__name__)


class SafeWebSearchWrapper:
    """
    Wrapper around Tavily web search that provides graceful fallback
    if TAVILY_API_KEY is not configured or an error occurs during invocation.
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.settings = get_settings()
        api_key = self.settings.tavily_api_key or os.getenv("TAVILY_API_KEY")

        if api_key:
            os.environ["TAVILY_API_KEY"] = api_key
            try:
                from langchain_tavily import TavilySearch
                self.tool = TavilySearch(max_results=self.k)
            except Exception:
                from langchain_community.tools.tavily_search import TavilySearchResults
                self.tool = TavilySearchResults(k=self.k)
            self.available = True
        else:
            logger.warning("TAVILY_API_KEY not configured. Web search will return fallback notice.")
            self.tool = None
            self.available = False

    def invoke(self, input_data: Dict[str, Any] | str) -> List[Dict[str, str]]:
        query = input_data["query"] if isinstance(input_data, dict) else str(input_data)
        if self.tool and self.available:
            try:
                results = self.tool.invoke({"query": query})
                if isinstance(results, list):
                    return results
                return [{"content": str(results), "url": "web_search"}]
            except Exception as exc:
                logger.error("Error executing Tavily search: %s", exc)
                return [
                    {
                        "content": f"[Web search for '{query}' encountered an error: {exc}]",
                        "url": "error",
                    }
                ]
        return [
            {
                "content": f"[Web search fallback: No TAVILY_API_KEY provided. Query '{query}' could not be searched on the web.]",
                "url": "mock",
            }
        ]


def get_web_search_tool(k: int | None = None) -> SafeWebSearchWrapper:
    """Returns the web search tool wrapper."""
    settings = get_settings()
    num_results = k if k is not None else settings.tavily_k
    return SafeWebSearchWrapper(k=num_results)
