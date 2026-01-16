# 🐾 PawAid ai

AI-powered first-aid assistant for pet emergencies (cats & dogs)

## What it does
Provides evidence-based first-aid guidance by retrieving relevant information 
from veterinary resources using RAG (Retrieval-Augmented Generation).

## Tech Stack
- RAG Pipeline: LangChain
- Vector DB: ChromaDB
- LLM: GPT-4
- Frontend: Streamlit

## Project Status
🚧 In Development - Building document ingestion pipeline

## Setup

### Prerequisites
- Conda or Miniconda installed
- OpenAI API key

### Installation

1. Clone the repository
```bash
git clone https://github.com/artuguen28/PawAid-AI.git
cd PawAid-Copilot
```

2. Create conda environment
```bash
conda env create -f environment.yml
conda activate pawaid
```

3. Set up environment variables
```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

4. Validate setup
```bash
python scripts/validate_setup.py
```
