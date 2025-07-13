import concurrent.futures
from typing import Any, Dict, List
from serpapi import GoogleSearch
from config import SERPAPI_KEY

class WebSearchTool:
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        queries = input_data.get("queries", [])
        docs = []
        seen_urls = set()

        def search(query):
            params = {
                "engine": "google",
                "q": query,
                "api_key": SERPAPI_KEY,
                "num": "10",
                "safe": "active",
                "hl": "en",
                "gl": "us"
            }
            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                return results.get("organic_results", [])
            except Exception as e:
                print(f"[WebSearchTool] Error searching query '{query}': {e}")
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            all_results = executor.map(search, queries)

        for result_list in all_results:
            for item in result_list:
                url = item.get("link")
                title = item.get("title")
                if url and url not in seen_urls:
                    docs.append({"title": title, "url": url})
                    seen_urls.add(url)

        return {"docs": docs}
