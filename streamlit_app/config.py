"""
Configuration Module
Loads environment variables and application settings
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# API Keys
OPENAI_KEY = os.getenv("OPENAI_KEY")
OPENROUTER_KEY = os.getenv("OPEN_ROUTER_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")

# Neo4j Configuration
def load_neo4j_config():
    """Load Neo4j configuration from config.txt file"""
    config = {}
    config_path = "config.txt"

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    config[key] = value

    return config

NEO4J_CONFIG = load_neo4j_config()

# Embedding Models Available
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "index_name": "playerEmbeddingTextIndexL6V2",
        "property_name": "embedding_text_l6_v2"
    },
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "index_name": "playerEmbeddingTextIndexMpnetV2",
        "property_name": "embedding_text_mpnet_v2"
    }
}

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
