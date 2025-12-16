# Fantasy Premier League Graph-RAG Assistant

A modular Streamlit application for querying Fantasy Premier League data using Graph-RAG (Retrieval-Augmented Generation).

## Project Structure

```
streamlit_app/
├── __init__.py              # Package initialization
├── config.py                # Configuration and environment variables
├── database.py              # Neo4j connection and operations
├── intent_queries.py        # Intent categories and Cypher query templates
├── intent_classifier.py     # LLM-based intent classification
├── entity_extractor.py      # Entity extraction and grounding
├── query_builder.py         # Cypher query population
├── retrieval.py             # Context retrieval (baseline + embeddings)
├── llm_service.py           # LLM generation functions (OpenAI, Gemini, OpenRouter)
├── visualization.py         # Graph visualization
├── initialize_kg.py         # Knowledge graph initialization script
└── app.py                   # Main Streamlit UI
```

## Features

### Input Processing (OpenAI-Powered with Structured Outputs)
- **Intent Classification**: Uses OpenAI GPT-5-mini with JSON mode to classify queries into 10 intent categories
  - Enforces structured JSON output: `{"category": "player_season_stats"}`
  - Validates against predefined categories, returns "Unknown" if no match
  - Temperature 0.1 for consistent, reliable classification
- **Smart Entity Extraction**: Uses OpenAI GPT-5-mini with JSON mode and enhanced prompting
  - Structured JSON output with all entity types: players, teams, positions, seasons, gameweeks, fixtures, statistics, time_references
  - **Context-aware**: Understands Fantasy Premier League domain and terminology
  - **Handles variations**: "goalkeepers"→"goalkeeper", "attackers"→"forward", "Man City"→"Manchester City"
  - **Comprehensive examples**: 8 examples covering edge cases (plurals, synonyms, abbreviations)
  - Eliminates manual parsing, ensures consistent format
  - Entities are then grounded against knowledge graph using fuzzy string matching

### Retrieval Methods (Optimized & Conditional)
- **Baseline**: Structured Cypher queries only
  - Only executes intent classification and query building when needed
  - Displays: Cypher query, knowledge graph visualization, structured data
- **Embeddings**: Semantic similarity using pre-computed embeddings
  - Only extracts entities for player names, skips intent classification
  - Returns full player node properties from Neo4j
  - Displays: Similar players table with names, similarity scores, and all node properties
- **Both**: Hybrid approach combining both methods
  - Executes both baseline and embeddings retrieval
  - Displays: All visualizations from both methods

**Performance Optimization**: The retrieval system now checks the selected method first and only executes the necessary processing, avoiding redundant LLM calls and database queries.

### Embedding Models
- **all-MiniLM-L6-v2**: Faster, 384 dimensions
- **all-mpnet-base-v2**: More accurate, 768 dimensions

Both models return complete player node properties including all statistics, positions, and metadata.

### LLM Models (for Answer Generation)
- **OpenAI GPT 5.1**
- **Google Gemini 2.5 Flash**
- **Meta LLama 3.3 70B** (via OpenRouter)

### Additional Features
- **Conditional Visualizations**: UI adapts based on retrieval method selected
  - Baseline/Both: Shows Cypher query + knowledge graph + structured data
  - Embeddings/Both: Shows semantic similarity results with full node properties
- Interactive knowledge graph visualization
- Real-time model comparison across all 3 LLMs
- Detailed metrics (response time, tokens, cost)
- Dynamic data display for all retrieved fields

## Installation

1. Install required packages:
```bash
pip install streamlit neo4j pandas python-dotenv openai langchain-openai langchain-google-genai networkx plotly
```

2. Set up environment variables in `.env`:
```
OPENAI_KEY=your_openai_key
OPEN_ROUTER_KEY=your_openrouter_key
GEMINI_KEY=your_gemini_key
HUGGING_FACE_TOKEN=your_huggingface_token
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

3. Create `config.txt` with Neo4j credentials:
```
URI=neo4j+s://your_neo4j_uri
USERNAME=your_username
PASSWORD=your_password
```

## Usage

### From Root Directory
```bash
streamlit run app_main.py
```

### From Streamlit App Directory
```bash
cd streamlit_app
streamlit run app.py
```

## Module Documentation

### config.py
Loads environment variables and defines application settings including available embedding models.

### database.py
Handles Neo4j database connections using a singleton pattern. Provides functions for:
- Getting knowledge graph entities (cached)
- Retrieving similar players by embedding with full node properties
  - Returns tuples: `(player_name, similarity_score, node)`
  - Node contains all player properties from Neo4j

### initialize_kg.py
Standalone script to initialize the Neo4j knowledge graph with FPL data and embeddings.

**Steps performed:**
1. Loads FPL data from CSV into Neo4j (batch processing: 1000 rows per batch)
2. Creates nodes (Season, Gameweek, Fixture, Team, Player, Position)
3. Creates relationships between nodes
4. Builds textual descriptions for each player
5. Computes player embeddings using two models:
   - sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
   - sentence-transformers/all-mpnet-base-v2 (768 dimensions)
6. Writes embeddings to Neo4j Player nodes
7. Creates vector indexes for semantic similarity search
   - playerEmbeddingTextIndexL6V2 (384 dimensions, cosine similarity)
   - playerEmbeddingTextIndexMpnetV2 (768 dimensions, cosine similarity)

**Usage:**
```bash
cd streamlit_app
python initialize_kg.py
```

### intent_queries.py
Defines the 10 intent categories for FPL queries:
1. player_basic_info
2. player_season_stats
3. player_gw_stats
4. fixture_details
5. team_fixtures
6. team_players
7. top_players_position
8. compare_players
9. gameweek_summary
10. player_vs_opponent

### intent_classifier.py
Uses OpenAI GPT-4o-mini with JSON mode to classify user queries into predefined intent categories.

**Key Functions:**
- `call_llm_for_classification(user_input)`: Calls OpenAI with structured JSON output
  - Uses `response_format: {"type": "json_object"}` to enforce JSON
  - Returns category name after validation against INTENTION_CATEGORIES
  - Returns "Unknown" if category doesn't match or JSON parsing fails
- `classify_intent(user_input)`: Main classification function
  - Returns tuple: (category_name, cypher_query_template)

### entity_extractor.py
Extracts entities from user queries using OpenAI GPT-4o-mini with JSON mode and grounds them against the knowledge graph.

**Key Functions:**
- `call_llm_for_entity_extraction(user_input)`: Calls OpenAI with enhanced structured JSON output
  - Uses `response_format: {"type": "json_object"}` to enforce JSON
  - **Smart extraction with context understanding**: Explains FPL domain to the model
  - **Handles variations and synonyms**:
    - Recognizes "goalkeepers", "keepers", "GK" → extracts "goalkeeper"
    - Recognizes "attackers", "strikers", "forwards" → extracts "forward"
    - Recognizes team abbreviations: "Man City", "Spurs", etc.
  - **8 comprehensive examples** covering edge cases (plural forms, abbreviations, complex queries)
  - Returns structured dict with all entity types: players, teams, positions, seasons, gameweeks, statistics, time_references
  - Validates JSON structure and ensures all required keys exist
- `extract_entities(user_input)`: Main extraction and grounding function
  - Calls `call_llm_for_entity_extraction()` for structured extraction
  - Grounds entities against knowledge graph using fuzzy string matching (difflib)
  - Returns tuple: (extracted_entities, grounded_entities)

### query_builder.py
Populates Cypher query templates with extracted entities. Validates required parameters and handles optional parameters intelligently.

**Key Features:**
- Extracts parameter placeholders from query templates ($player, $season, etc.)
- Maps parameters to grounded entity types
- Validates that all required (non-OPTIONAL) parameters are present
- **Smart default handling for optional parameters**:
  - If `$season` is in an OPTIONAL MATCH but unassigned → defaults to '22/23' (latest season)
  - Other optional parameters → default to None (NULL in Neo4j)
  - Prevents unassigned variable errors in Cypher queries
- Returns None if required parameters are missing (fixes the TypeError bug)

### retrieval.py
Orchestrates context retrieval using three methods with optimized conditional processing.

**Key Features:**
- **Conditional Processing**: Checks retrieval method first to avoid redundant operations
  - `baseline`: Only runs intent classification, entity extraction, and Cypher queries
  - `embeddings`: Only runs entity extraction (for player names), skips intent classification
  - `both`: Runs both baseline and embeddings retrieval
- **Performance Optimization**: Eliminates unnecessary LLM calls and database queries based on selected method
- **Enhanced Return Values**: Returns 5-tuple: `(unified_context, normalized_baseline, embeddings_context, cypher_query, query_params)`
  - `unified_context`: Combined results from all retrieval methods
  - `normalized_baseline`: Structured results from Cypher queries
  - `embeddings_context`: List of `(player_name, similarity_score, node)` tuples
  - `cypher_query`: The executed Cypher query (or None)
  - `query_params`: Parameters used in the query (or None)

**Functions:**
- `normalize_baseline_result(result_list)`: Converts Neo4j results into unified format
  - Handles Player, Team, Position, Fixture, Season, Gameweek nodes
  - Extracts relationships and aggregated statistics
- `get_context_enhanced(question, retrieval_method, embedding_model)`: Main retrieval orchestrator
  - Conditionally executes baseline and/or embeddings retrieval
  - Returns comprehensive context for LLM generation

### llm_service.py
Provides unified interface for three LLM providers with consistent metrics and user-friendly responses.

**Key Features:**
- **Conversational Tone**: Responses are friendly and natural, like chatting with a fellow FPL fan
- **Abstracted Implementation**: Never mentions "context", "database", or technical details to users
- **Smart Response Handling**:
  - When data available: Presents answers directly with clear stats and details
  - When data missing: Simply says "I don't have that information" without technical explanations
- **Technical Metadata Filtering**: Strictly hides all technical details from responses
  - Never reveals internal IDs (player_element, element_code, etc.)
  - Never mentions "entries", "records", "duplicates", or data structure
  - Never explains data inconsistencies or how data is stored
  - Extracts ONLY FPL-relevant information to answer the question
- **Natural Knowledge Presentation**: Answers as if naturally knowledgeable, not as a data lookup system
  - Never says "from the data", "based on what I see", "from what's provided"
  - Never implies the user shared any data or context
  - Answers confidently and directly like an FPL expert would
- **Consistent Experience**: All three models (OpenAI, Gemini, OpenRouter) use the same friendly system prompt
- **Trivia-Style Answers**: Enthusiastic, engaging responses focused on FPL knowledge, answering only what was asked

### visualization.py
Creates interactive Plotly visualizations of the knowledge graph.

### app.py
Main Streamlit application with all UI components.

## Improvements

### Performance & Optimization
- **Conditional Retrieval Processing**: Retrieval system now checks the selected method first
  - **Embeddings mode**: Skips intent classification and Cypher query building entirely
  - **Baseline mode**: Skips embedding retrieval
  - **Both mode**: Executes both methods efficiently
  - Eliminates redundant LLM calls and database queries, improving response time
  - Allows UI to display embeddings data independently from baseline data
  - Full node properties included in embeddings results

### Data Handling
- **Optional Parameter Handling**: Unassigned optional parameters are now given smart defaults
  - Season parameters default to '22/23' (latest season)
  - Other optional parameters default to None (NULL)
  - Prevents Cypher query errors from unassigned variables
- **Fixtures Entity Type**: Added fixture number extraction to entity extraction pipeline
  - Recognizes patterns like "Fixture 12", "fixture 162", "match 45"
  - Extracts numeric fixture IDs for query population
- **Full Node Properties in Embeddings**: Database now returns complete player nodes with all properties
  - All player statistics, positions, and metadata available for context

### Input Processing
- **Structured Outputs with Enhanced Prompting**: Intent classification and entity extraction now use OpenAI's JSON mode
  - Eliminates manual text parsing and potential parsing errors
  - Ensures consistent, reliable output format
  - Validates output structure before use
  - **Smart entity extraction**: Recognizes variations (plural forms, synonyms, abbreviations)
  - **Context-aware**: Prompts explain FPL domain for better understanding
  - **Comprehensive examples**: 8 diverse examples covering edge cases

### User Experience
- **Conditional Visualizations**: UI adapts based on retrieval method
  - Baseline: Shows Cypher query + knowledge graph + structured data
  - Embeddings: Shows semantic similarity results with full node properties
  - Both: Shows all visualizations from both methods
- **Dynamic Data Display**: Shows all retrieved fields instead of hardcoded subset
  - Automatically detects and displays all fields in baseline results
  - Proper handling of different data types (dict, list, primitives)
- **User-Friendly Responses**: LLM responses are now conversational and trivia-style
  - Removed technical jargon like "context provided" or "empty list"
  - When data found: Presents answers directly and clearly
  - When data missing: Simply says "I don't have that information" without technical explanations
  - **Hides all technical metadata**: Never reveals internal IDs, database fields, duplicate entries, or data structure
  - **Answers only what was asked**: Extracts only FPL-relevant information, ignores technical details
  - **Natural knowledge presentation**: Never says "from the data you've shared" or implies user provided context
  - **Strict context adherence**: Uses ONLY provided data, no external knowledge or assumptions
  - Answers as a knowledgeable FPL fan, not as a database query system
  - Natural, enthusiastic tone like chatting with a fellow FPL fan

### Code Quality
- **Modular Structure**: Code is now properly organized into logical modules
- **Proper Neo4j Node Handling**: Graph visualization correctly extracts node names from Neo4j dictionaries
  - Handles Player, Team, Position, Fixture, Gameweek, Season nodes

## Future Enhancements

- Add more query intents
- Implement caching for LLM responses
- Add query history
- Add more data sources
- Add more visualization options
- Improve embedding choices
