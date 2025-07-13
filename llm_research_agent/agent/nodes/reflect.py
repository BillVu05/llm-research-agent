from typing import Any, Dict

class Reflect:
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        docs = input_data.get("docs", [])
        queries = input_data.get("queries", [])
        need_more = len(docs) < 6
        refined = [q + " (refined)" for q in queries[:3]] if need_more else []
        return {"need_more": need_more, "new_queries": refined, "docs": docs, "queries": queries}
