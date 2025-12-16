"""
Database Module
Handles Neo4j database connection and operations
"""

from neo4j import GraphDatabase
import streamlit as st
from config import NEO4J_CONFIG

class Neo4jConnection:
    """Singleton class for Neo4j database connection"""

    _instance = None
    _driver = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jConnection, cls).__new__(cls)
        return cls._instance

    def get_driver(self):
        """Get or create Neo4j driver instance"""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                NEO4J_CONFIG["URI"],
                auth=(NEO4J_CONFIG["USERNAME"], NEO4J_CONFIG["PASSWORD"])
            )
        return self._driver

    def close(self):
        """Close Neo4j driver connection"""
        if self._driver is not None:
            self._driver.close()
            self._driver = None


@st.cache_data(ttl=3600)
def get_kg_entities():
    """
    Retrieve all entity values from the knowledge graph for entity grounding.

    Returns:
        dict: Dictionary containing lists of players, teams, positions, seasons, and gameweeks
    """
    driver = Neo4jConnection().get_driver()

    with driver.session() as session:
        # Get all players
        players = session.run("MATCH (p:Player) RETURN p.player_name as name").data()
        player_names = [p['name'] for p in players if p['name']]

        # Get all teams
        teams = session.run("MATCH (t:Team) RETURN t.name as name").data()
        team_names = [t['name'] for t in teams if t['name']]

        # Get all positions
        positions = session.run("MATCH (pos:Position) RETURN pos.name as name").data()
        position_names = [pos['name'] for pos in positions if pos['name']]

        # Get all seasons
        seasons = session.run("MATCH (s:Season) RETURN s.season_name as name").data()
        season_names = [s['name'] for s in seasons if s['name']]

        # Get gameweek range
        gameweeks = session.run("MATCH (g:Gameweek) RETURN DISTINCT g.GW_number as gw ORDER BY gw").data()
        gw_numbers = [gw['gw'] for gw in gameweeks if gw['gw']]

    return {
        'players': player_names,
        'teams': team_names,
        'positions': position_names,
        'seasons': season_names,
        'gameweeks': gw_numbers
    }


def get_similar_players_by_embedding(player_name, embedding_model_key="all-mpnet-base-v2", top_k=5):
    """
    Retrieve similar players using pre-computed embeddings from Neo4j.

    Args:
        player_name: Name of the player to find similar players for
        embedding_model_key: Key for the embedding model to use ("all-MiniLM-L6-v2" or "all-mpnet-base-v2")
        top_k: Number of similar players to return

    Returns:
        list: List of tuples (player_name, similarity_score)
    """
    from config import EMBEDDING_MODELS

    model_info = EMBEDDING_MODELS[embedding_model_key]
    embedding_prop = model_info["property_name"]
    index_name = model_info["index_name"]

    query = f"""
    MATCH (p:Player {{player_name: $name}})
    WITH p.{embedding_prop} AS query_vec
    CALL db.index.vector.queryNodes('{index_name}', $top_k, query_vec)
    YIELD node, score
    RETURN node.player_name AS similar_player, score, node
    ORDER BY score DESC
    """

    driver = Neo4jConnection().get_driver()

    with driver.session() as session:
        results = session.run(query, name=player_name, top_k=top_k)
        return [(r["similar_player"], r["score"], r["node"]) for r in results]
