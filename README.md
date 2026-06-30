# Conversational RAG Chatbot for AI Student Impact Dataset

This project builds a Retrieval-Augmented Generation (RAG) chatbot over the `ai_student_impact_dataset.csv` dataset.

## Features
- Loads the CSV dataset as LangChain documents
- Splits text into chunks with `RecursiveCharacterTextSplitter`
- Builds a FAISS vector store and vector retriever
- Uses a chat prompt template with `MessagesPlaceholder` for history
- Maintains and trims conversation history
- Answers only from retrieved context and says "I don't know" when unsupported

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. This app uses Hugging Face embeddings by default and does not require an OpenAI API key.

   To use OpenAI embeddings instead, set:
   ```bash
   setx OPENAI_API_KEY "your_api_key_here"
   setx USE_OPENAI "1"
   ```

## Run

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

Run the CLI chatbot:
```bash
python -m src.ai_student_impact.cli --dataset path/to/ai_student_impact_dataset.csv --max-rows 100
```

## Deployment
This app can be deployed for free:
- Streamlit Community Cloud: push this repository to GitHub and connect the repo. Streamlit will launch `streamlit_app.py` automatically.
- Hugging Face Spaces: use a Python Space with `requirements.txt` and `streamlit_app.py`.

## Notes
- The app uses OpenAI generation when `OPENAI_API_KEY` is set.
- If you prefer OpenAI-only behavior, ensure `OPENAI_API_KEY` and `OPENAI_MODEL` are configured.
- For local use, set `DATASET_PATH` to your CSV file.
