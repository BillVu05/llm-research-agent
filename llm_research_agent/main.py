import argparse
import json
from agent.pipeline import build_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, required=False)
    parser.add_argument("question", nargs="?", help="Research question (positional for Docker)")
    args = parser.parse_args()

    topic = args.topic or args.question
    if not topic:
        parser.error("Please provide a research topic/question.")

    pipeline = build_pipeline()
    result = pipeline.run({"topic": topic})

    # Append citation IDs in answer if missing (optional)
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    ids = [str(c.get("id")) for c in citations if "id" in c]
    if ids and not any(f"[{id}]" in answer for id in ids):
        answer = f"{answer.strip()} [{', '.join(ids)}]"

    clean_output = {
        "answer": answer,
        "citations": citations
    }
    print(json.dumps(clean_output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
