"""
Retrieval Module
Orchestrates context retrieval using baseline Cypher queries and/or embeddings

Supports three retrieval methods:
1. Baseline: Structured Cypher queries only
2. Embeddings: Semantic similarity using pre-computed embeddings
3. Both: Hybrid approach combining structured and semantic retrieval
"""

from intent_classifier import classify_intent
from entity_extractor import extract_entities
from query_builder import populate_query
from database import Neo4jConnection, get_similar_players_by_embedding
from neo4j.graph import Node, Relationship


def normalize_baseline_result(result_list):
    """
    Convert Neo4j baseline output into a unified format.

    Handles different node types (Player, Team, Position, Fixture),
    relationships, and aggregated statistics.

    Args:
        result_list: List of Neo4j query results

    Returns:
        list: Normalized results in a consistent dictionary format
    """
    unified = []

    for record in result_list:
        item = {
            "type": "structured",
            "players": [],
            "teams": [],
            "positions": [],
            "fixtures": [],
            "relationships": [],
            "aggregated_stats": {}
        }

        for key, value in record.items():
            # Player node
            if key == "p" or key == "player" or key.startswith("player"):
                if isinstance(value, dict):  # Neo4j node
                    item["players"].append({
                        "player_name": value.get("player_name", "Unknown"),
                        "player_code": value.get("code"),
                        "other_props": {k: v for k, v in value.items()
                                        if k not in ["player_name", "code"]}
                    })
                else:  # Aggregated name string
                    item["players"].append({"player_name": value})

            # Team node
            elif key == "t":
                item["teams"].append({
                    "team_name": value.get("name", "Unknown") if isinstance(value, dict) else str(value)
                })

            # Position node
            elif key == "pos":
                item["positions"].append({
                    "position_name": value.get("name", "Unknown") if isinstance(value, dict) else str(value)
                })

            # Fixture node
            elif key == "f":
                if isinstance(value, dict):
                    item["fixtures"].append({
                        "fixture_number": value.get("fixture_number"),
                        "season": value.get("season"),
                        "kickoff_time": value.get("kickoff_time"),
                        "other_props": {k: v for k, v in value.items()
                                        if k not in ["fixture_number", "season", "kickoff_time"]}
                    })

            # Relationship type (from r which returns as tuple)
            elif key == "r" and isinstance(value, tuple):
                # Store relationship type from tuple (middle element)
                print("rel", value)
                item["_temp_rel_type"] = value[1] if len(value) >= 2 else "UNKNOWN"
                item["_startNode"] = value[0]
                item["_endNode"] = value[2]

            # Relationship properties
            elif key == "r_props" and isinstance(value, dict):
                print("rprops", value)
                item["relationships"].append({
                    "from": item.get("_startNode", "UNKNOWN"),
                    "to": item.get("_endNode", "UNKNOWN"),
                    "type": item.get("_temp_rel_type", "UNKNOWN"),
                    "properties": value
                })
                # Clean up temporary storage
                item.pop("_temp_rel_type", None)
                item.pop("_startNode", None)
                item.pop("_endNode", None)

            # Aggregated stats
            elif key in ["assists", "minutes", "avg_form", "total_points", "goals",
                         "points", "p1_points", "p2_points"]:
                item["aggregated_stats"][key] = value

            # Player comparison
            elif key in ["player1", "player2"]:
                item["players"].append({"player_name": value})

            else:
                # Fallback: store in aggregated_stats
                print("IT SHOULD COME HERE", key, " value ", value)
                item["aggregated_stats"][key] = value
        unified.append(item)

    return unified


def get_context_enhanced(question, retrieval_method="both", embedding_model="all-mpnet-base-v2"):
    """
    Enhanced context retrieval function with multiple retrieval strategies.

    Args:
        question: User's natural language question
        retrieval_method: "baseline", "embeddings", or "both"
        embedding_model: Which embedding model to use ("all-MiniLM-L6-v2" or "all-mpnet-base-v2")

    Returns:
        tuple: (unified_context, normalized_baseline, embeddings_context, cypher_query, query_params)
            - unified_context: Combined results from all retrieval methods
            - normalized_baseline: Structured results from Cypher queries
            - embeddings_context: Semantic similarity results
            - cypher_query: The Cypher query that was executed
            - query_params: Parameters used in the query
    """
    # Initialize return values
    normalized_baseline = []
    embeddings_context = []
    cypher_query = None
    query_params = None
    grounded_entities = {}

    # Step 1: Baseline retrieval (only if needed)
    if retrieval_method in ["baseline", "both"]:
        # Extract structured query information
        category, query = classify_intent(question)
        print(f"Classified Intent: {category}\nGenerated Query: {query}")
        _, grounded_entities = extract_entities(question)
        print(f"Grounded Entities: {grounded_entities}")
        query_params = populate_query(query, grounded_entities)
        print(f"Populated Query Params: {query_params}")

        # Run baseline Cypher query (if params available)
        cypher_query = query if query else "No query generated"

        if query_params is not None and query is not None:
            driver = Neo4jConnection().get_driver()
            with driver.session() as session:
                result1 = session.run(query, query_params)
                baseline = result1.data()
            normalized_baseline = normalize_baseline_result(baseline)

    # Step 2: Embeddings retrieval (only if needed)
    if retrieval_method in ["embeddings", "both"]:

        # Get embedding-based contexts
        context = get_similar_players_by_embedding(
            question,
            embedding_model_key=embedding_model,
            top_k=5
        )
        embeddings_context.extend(context)

    # Step 3: Combine results
    unified_context = []

    # Add embedding-based results
    for player_name, score, node in embeddings_context:
        unified_context.append({
            "type": "semantic",
            "player_name": player_name,
            "similarity_score": score,
            "node_properties": dict(node)
        })

    unified_context.extend(normalized_baseline)

    return unified_context, normalized_baseline, embeddings_context, cypher_query, query_params
