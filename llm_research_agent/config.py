import os
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
