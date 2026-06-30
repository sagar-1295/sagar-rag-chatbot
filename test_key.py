import os
import sys

try:
    import openai
except ImportError:
    print("Missing dependency: install the OpenAI Python package first (pip install openai)")
    sys.exit(1)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

if not OPENAI_API_KEY:
    print("OPENAI_API_KEY is not set. Please export it before running this test.")
    sys.exit(1)

openai.api_key = OPENAI_API_KEY

print(f"Testing OpenAI key with model: {OPENAI_MODEL}")

try:
    response = openai.responses.create(
        model=OPENAI_MODEL,
        input="Say hello in one word.",
        max_output_tokens=1,
    )
    print("OpenAI key is valid.")
    print("Response:", response.output[0].content[0].text.strip())
except Exception as exc:
    print("API call failed:", exc)
    sys.exit(1)
