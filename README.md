# Prototype of a GPT based QA system for interactive user manuals

## Usage

### Setup python environment
    pip3 install langchain openai tiktoken faiss-cpu unstructured flask

### Export OpenAI API key
    export OPENAI_API_KEY="your_api_key"

### Pre-process source PDF
    python3 create_vector_store.py

### Run demo app
    python3 app.py
