# Fantasy Football Predictor

Graph-RAG assistant for Fantasy Premier League (FPL) analysis using:
- Neo4j knowledge graph for structured football data
- Embedding-based semantic retrieval for similar player search
- Multiple LLMs for natural-language answers in a Streamlit UI

This repository contains data processing notebooks, graph initialization scripts, and two Streamlit app entrypoints (legacy and modular).

## What This Project Does

You can ask questions such as:
- "How did Salah perform in GW 5?"
- "Compare Son and Rashford this season"
- "Top scoring midfielders in 22/23"

The app retrieves relevant context from the knowledge graph using one of three retrieval modes:
- `baseline`: Cypher query retrieval
- `embeddings`: vector similarity retrieval
- `both`: hybrid retrieval (recommended)

## Repository Layout

Key files and folders:
- `streamlit_app/`: modular application package (recommended app)
- `app_main.py`: main root entrypoint that runs the modular app
- `streamlit_app.py`: legacy monolithic Streamlit app
- `streamlit_app/initialize_kg.py`: script to build the Neo4j graph and vector indexes
- `fpl_graph_final.csv`: primary dataset used for graph initialization
- `docker-compose.yml`: local Neo4j container setup
- `requirements.txt`: Python dependencies

## Requirements

- Python 3.10+
- Neo4j 5+ (local or hosted)
- API keys for one or more LLM providers

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the repository root:

```env
OPENAI_KEY=your_openai_key
OPEN_ROUTER_KEY=your_openrouter_key
GEMINI_KEY=your_gemini_key
```

Create a `config.txt` file in the repository root:

```txt
URI=neo4j+s://your_neo4j_uri
USERNAME=neo4j
PASSWORD=your_password
```

Notes:
- If you run the modular app from `streamlit_app/`, it reads `config.txt` from that working directory.
- Keep `config.txt` available in the directory where you launch the app/script.

## Start Neo4j (Optional via Docker)

If you want local Neo4j:

```bash
docker compose up -d
```

Default values in `docker-compose.yml` expose:
- HTTP: `7474`
- Bolt: `7687`

## Initialize the Knowledge Graph

Before using embeddings retrieval, initialize graph data and vector indexes:

```bash
cd streamlit_app
python initialize_kg.py
```

This step:
- Loads `fpl_graph_final.csv` into Neo4j
- Creates nodes/relationships for seasons, gameweeks, fixtures, teams, players
- Computes embeddings using:
	- `sentence-transformers/all-MiniLM-L6-v2`
	- `sentence-transformers/all-mpnet-base-v2`
- Creates vector indexes for semantic search

## Run the App

Recommended (modular app):

```bash
streamlit run app_main.py
```

Alternative (legacy app):

```bash
streamlit run streamlit_app.py
```

Then open the local Streamlit URL shown in your terminal.

## Retrieval and Model Options

In the sidebar, you can configure:
- Retrieval mode: `baseline`, `embeddings`, or `both`
- Embedding model:
	- `all-MiniLM-L6-v2` (faster, 384 dims)
	- `all-mpnet-base-v2` (typically stronger quality, 768 dims)
- LLM provider/model for answer generation
- Optional 3-model comparison mode

## Troubleshooting

- Neo4j connection errors:
	- Verify `URI`, `USERNAME`, `PASSWORD` in `config.txt`
	- Confirm Neo4j is running and reachable
- Missing API key errors:
	- Confirm `.env` exists and keys are set correctly
- Empty embeddings results:
	- Run `streamlit_app/initialize_kg.py` to generate vectors and indexes
- Slow first query:
	- Sentence-transformer models may take time to load on first use

## Development Notes

- Notebooks (`*.ipynb`) in this repo contain data prep, modeling, and experimentation.
- `streamlit_app/` is the clean, modular codebase for app development.
- Root-level scripts are useful for quick experiments and backward compatibility.

## License

This project is licensed under the terms in `LICENSE`.
