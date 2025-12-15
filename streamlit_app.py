"""
Fantasy Premier League Graph-RAG Assistant
Auto-generated from final.ipynb

This app uses pre-computed embeddings stored in Neo4j.
Run the notebook first to create embeddings.
"""

import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import re
from difflib import get_close_matches
import time
import networkx as nx
import plotly.graph_objects as go

st.set_page_config(
    page_title="Fantasy Premier League Assistant",
    page_icon="⚽",
    layout="wide"
)


# ===== Cell 1 =====
from neo4j import GraphDatabase
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import tqdm


# ===== Cell 2 =====
intention_categories = ["player_basic_info", "player_season_stats", "player_gw_stats", 
                        "fixture_details", "team_fixtures", "team_players", "top_players_position","compare_players","gameweek_summary","player_vs_opponent"
]

intention_map = {
    "player_basic_info": """
MATCH (p:Player {player_name: $player})
OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
OPTIONAL MATCH (p)-[:PLAYS_FOR {season: $season}]->(t:Team)
RETURN p, pos, t;""",
    "player_season_stats": """
MATCH (p:Player {player_name: $player})-[r:PLAYED_IN]->(f:Fixture)
MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek)<-[:HAS_GW]-(s:Season {season_name: $season})
RETURN p.player_name AS player,
       SUM(r.total_points) AS total_points,
       SUM(r.goals_scored) AS goals,
       SUM(r.assists) AS assists,
       SUM(r.minutes) AS minutes,
       AVG(r.form) AS avg_form;
""",
    "player_gw_stats":"""
MATCH (p:Player {player_name: $player})-[r:PLAYED_IN]->(f:Fixture)
MATCH (g:Gameweek {season: $season, GW_number: $gw})-[:HAS_FIXTURE]->(f)
RETURN p, r, f;
""",
    "fixture_details":"""
MATCH (p:Player {player_name: $player})-[r:PLAYED_IN]->(f:Fixture)
MATCH (g:Gameweek {season: $season, GW_number: $gw})-[:HAS_FIXTURE]->(f)
RETURN p, r, f;
""",
    "team_fixtures":"""
MATCH (t:Team {name: $team})
MATCH (f:Fixture)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)
RETURN f ORDER BY f.kickoff_time;
""",
    "team_players":"""
MATCH (t:Team {name: $team})
MATCH (p:Player)-[:PLAYS_FOR {season: $season}]->(t)
RETURN p;
""",
    "top_players_position":"""
MATCH (p:Player)-[:PLAYS_AS]->(:Position {name: $position})
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek)<-[:HAS_GW]-(s:Season {season_name: $season})
RETURN p.player_name AS player, SUM(r.total_points) AS points
ORDER BY points DESC LIMIT $limit;
""",
    "compare_players":"""
MATCH (p1:Player {player_name: $player1})-[r1:PLAYED_IN]->(f1:Fixture)
MATCH (p2:Player {player_name: $player2})-[r2:PLAYED_IN]->(f2:Fixture)
RETURN p1.player_name AS player1, SUM(r1.total_points) AS p1_points,
       p2.player_name AS player2, SUM(r2.total_points) AS p2_points;
""",
    "gameweek_summary":"""
MATCH (g:Gameweek {season: $season, GW_number: $gw})-[:HAS_FIXTURE]->(f)
MATCH (p:Player)-[r:PLAYED_IN]->(f)
RETURN g, f, p, r
ORDER BY f.fixture_number;
""",
    "player_vs_opponent":"""
MATCH (p:Player {player_name: $player})-[r:PLAYED_IN]->(f:Fixture)
MATCH (p)-[:PLAYED_AGAINST]->(opp:Team {name: $opponent})
RETURN p.player_name AS player, opp.name AS opponent,
       SUM(r.goals_scored) AS goals,
       SUM(r.assists) AS assists,
       SUM(r.total_points) AS points;
"""
}


# ===== Cell 3 =====
from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv(override=True)

# Load API key from environment variable for security
api_key = os.getenv("OPEN_ROUTER_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

def call_open_router(prompt: str) -> str:
    completion = client.chat.completions.create(
        extra_body={},
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        max_tokens=1000  # Limit the response length to reduce cost
    )
    return completion.choices[0].message.content



# ===== Cell 4 =====
# We are going to use LLM-based classification
def classify_intent(user_input):
    # TODO: This should return the Cypher Queries (descriptions) associated with each intention Or the Retrieval methods to use.

    prompt = f"""
You are an intent classifier for a Fantasy Premier League (FPL) knowledge graph system.

Your task:
Given a user query, classify it into EXACTLY one of the following categories:
{', '.join(intention_categories)}

Category definitions (important):
- player_basic_info: Asking who a player is, their position, or their team.
- player_season_stats: Asking about a player's overall seasonal performance.
- player_gw_stats: Asking about a player's performance in a specific gameweek.
- fixture_details: Asking about a specific match, its teams, or players in it.
- team_fixtures: Asking about a team's upcoming or past fixtures.
- team_players: Asking which players belong to a team.
- top_players_position: Asking for ranking or best players in a position/season.
- compare_players: Comparing two players statistically.
- gameweek_summary: Asking about all fixtures or events in a specific GW.
- player_vs_opponent: Asking how a player performed against a specific team.

Examples:
User Input: "Show me Haaland's stats last season."
Category: player_season_stats

User Input: "How did Salah do in GW 5?"
Category: player_gw_stats

User Input: "Who plays for Arsenal this season?"
Category: team_players

User Input: "Which fixtures does Liverpool have next month?"
Category: team_fixtures

User Input: "Tell me which defender scored the most points last year."
Category: top_players_position

User Input: "Compare Son and Rashford this season."
Category: compare_players

User Input: "What happened in gameweek 10?"
Category: gameweek_summary

User Input: "How does Kane perform against Chelsea?"
Category: player_vs_opponent

User Input: "Who is Trent Alexander-Arnold?"
Category: player_basic_info

Now classify the user's input below.
Return ONLY the category name from the list above.

User Input: "{user_input}"
Category:
"""
    category = call_open_router(prompt).strip()
    if category not in intention_categories:
        print("Warning: LLM returned an unexpected category.")
        print(f"LLM Output: {category}")
        category = "Unknown"
    return category, intention_map.get(category)


# ===== Cell 5 =====
from neo4j import GraphDatabase

# Initialize Neo4j connection for entity grounding
config = {}
with open("config.txt", "r") as f:
    for line in f:
        key, value = line.strip().split("=", 1)
        config[key] = value

driver = GraphDatabase.driver(config["URI"], auth=(config["USERNAME"], config["PASSWORD"]))

def get_kg_entities():
    """
    Retrieve all entity values from the knowledge graph to ground entity extraction.
    Returns dictionaries of players, teams, positions, and seasons.
    """
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

# Cache KG entities for faster lookups
kg_entities = get_kg_entities()
print(f"Loaded {len(kg_entities['players'])} players, {len(kg_entities['teams'])} teams, "
      f"{len(kg_entities['positions'])} positions, {len(kg_entities['seasons'])} seasons")



# ===== Cell 6 =====
import re
from difflib import get_close_matches

def extract_entities(user_input):
    """
    Extract and ground entities from user input using the knowledge graph.
    Returns a structured dictionary with entity types and values validated against the KG.
    """
    
    # Step 1: Use LLM to identify potential entities and their types
    prompt = f"""Extract entities from the following fantasy football query. For each entity, identify its type.
Return the result in this exact format: EntityType: value1, value2
Available entity types: Player, Team, Position, Season, Gameweek, Statistic, TimeReference
Do not include any explanations or additional text.

Examples:
User Input: "Who is the top scoring midfielder this season?"
Player: 
Team: 
Position: midfielder
Season: this season
Gameweek: 
Statistic: top scoring
TimeReference: this season

User Input: "Find me a West Ham midfielder that scored the most points last season"
Player: 
Team: West Ham
Position: midfielder
Season: last season
Gameweek: 
Statistic: most points
TimeReference: last season

User Input: "How many goals did Salah score in gameweek 5?"
Player: Salah
Team: 
Position: 
Season: 
Gameweek: 5
Statistic: goals
TimeReference: gameweek 5

User Input: "{user_input}"
Player: 
Team: 
Position: 
Season: 
Gameweek: 
Statistic: 
TimeReference: 
"""
    
    llm_response = call_open_router(prompt).strip()
    print("llm_response: ",llm_response)
    # Step 2: Parse LLM response
    extracted = {
        'players': [],
        'teams': [],
        'positions': [],
        'seasons': [],
        'gameweeks': [],
        'statistics': [],
        'time_references': []
    }
    
    lines = llm_response.split('\n')
    for line in lines:
        if ':' in line:
            entity_type, values = line.split(':', 1)
            entity_type = entity_type.strip().lower()
            values = values.strip()
            
            if values and values.lower() not in ['none', 'n/a', '']:
                value_list = [v.strip() for v in values.split(',') if v.strip()]
                
                if 'player' in entity_type:
                    extracted['players'].extend(value_list)
                elif 'team' in entity_type:
                    extracted['teams'].extend(value_list)
                elif 'position' in entity_type:
                    extracted['positions'].extend(value_list)
                elif 'season' in entity_type:
                    extracted['seasons'].extend(value_list)
                elif 'gameweek' in entity_type:
                    extracted['gameweeks'].extend(value_list)
                elif 'statistic' in entity_type:
                    extracted['statistics'].extend(value_list)
                elif 'time' in entity_type:
                    extracted['time_references'].extend(value_list)
    
    # Step 3: Ground entities against the knowledge graph
    grounded_entities = {
        'players': [],
        'teams': [],
        'positions': [],
        'seasons': [],
        'gameweeks': [],
        'statistics': [],
        'time_references': extracted['time_references']
    }
    
    # Ground players
    for player in extracted['players']:
        matches = get_close_matches(player, kg_entities['players'], n=3, cutoff=0.2)
        if matches:
            grounded_entities['players'].append({
                'original': player,
                'grounded': matches[0],
                'alternatives': matches[1:] if len(matches) > 1 else []
            })
    
    # Ground teams
    for team in extracted['teams']:
        matches = get_close_matches(team, kg_entities['teams'], n=3, cutoff=0.2)
        if matches:
            grounded_entities['teams'].append({
                'original': team,
                'grounded': matches[0],
                'alternatives': matches[1:] if len(matches) > 1 else []
            })
    
    # Ground positions (normalize to KG format)
    position_mapping = {
        'goalkeeper': 'GK',
        'gk': 'GK',
        'defender': 'DEF',
        'def': 'DEF',
        'midfielder': 'MID',
        'mid': 'MID',
        'forward': 'FWD',
        'fwd': 'FWD',
        'striker': 'FWD',
        'attacker': 'FWD'
    }
    
    for position in extracted['positions']:
        position_lower = position.lower()
        if position_lower in position_mapping:
            mapped_pos = position_mapping[position_lower]
            if mapped_pos in kg_entities['positions']:
                grounded_entities['positions'].append({
                    'original': position,
                    'grounded': mapped_pos
                })
        else:
            matches = get_close_matches(position, kg_entities['positions'], n=1, cutoff=0.2)
            if matches:
                grounded_entities['positions'].append({
                    'original': position,
                    'grounded': matches[0]
                })
    
    # Ground seasons
    for season in extracted['seasons']:
        # Handle relative references
        if 'this' in season.lower() or 'current' in season.lower():
            latest_season = max(kg_entities['seasons']) if kg_entities['seasons'] else None
            if latest_season:
                grounded_entities['seasons'].append({
                    'original': season,
                    'grounded': latest_season,
                    'is_relative': True
                })
        elif 'last' in season.lower() or 'previous' in season.lower():
            sorted_seasons = sorted(kg_entities['seasons'], reverse=True)
            if len(sorted_seasons) > 1:
                grounded_entities['seasons'].append({
                    'original': season,
                    'grounded': sorted_seasons[1],
                    'is_relative': True
                })
        else:
            matches = get_close_matches(season, kg_entities['seasons'], n=1, cutoff=0.2)
            if matches:
                grounded_entities['seasons'].append({
                    'original': season,
                    'grounded': matches[0],
                    'is_relative': False
                })
    
    # Extract gameweek numbers
    for gw in extracted['gameweeks']:
        # Extract numeric value
        gw_match = re.search(r'\d+', gw)
        if gw_match:
            gw_num = int(gw_match.group())
            if gw_num in kg_entities['gameweeks']:
                grounded_entities['gameweeks'].append({
                    'original': gw,
                    'grounded': gw_num
                })
    
    # Keep statistics as-is (these are performance metrics)
    grounded_entities['statistics'] = extracted['statistics']
    
    return extracted, grounded_entities



# ===== Cell 7 =====
def populate_query(query, entities):
    """
    Populate a Cypher query template with extracted and grounded entities.
    
    Args:
        query (str): Cypher query template with parameter placeholders (e.g., $player, $season)
        entities (dict): Dictionary of grounded entities from extract_entities function
        
    Returns:
        tuple: (populated_query, parameters_dict)
            - populated_query: The original query (for Neo4j driver execution)
            - parameters_dict: Dictionary of parameters to pass to Neo4j
    """
    
    # Extract all parameter names from the query using regex
    # Matches $parameter_name patterns
    param_pattern = r'\$(\w+)'
    required_params = set(re.findall(param_pattern, query))
    
    # Initialize parameters dictionary
    parameters = {}
    
    # Mapping of parameter names to entity types
    param_to_entity_map = {
        'player': 'players',
        'player1': 'players',
        'player2': 'players',
        'team': 'teams',
        'opponent': 'teams',
        'position': 'positions',
        'season': 'seasons',
        'gw': 'gameweeks',
        'limit': 'statistics'  # Special case for LIMIT clauses
    }
    
    # Populate parameters based on extracted entities
    for param in required_params:
        entity_type = param_to_entity_map.get(param)
        
        if entity_type and entity_type in entities:
            entity_list = entities[entity_type]
            
            if entity_list:
                if param == 'player1' and len(entity_list) >= 1:
                    # For player comparisons, use first player
                    parameters[param] = entity_list[0]['grounded']
                elif param == 'player2' and len(entity_list) >= 2:
                    # For player comparisons, use second player
                    parameters[param] = entity_list[1]['grounded']
                elif param in ['player', 'team', 'opponent', 'position', 'season']:
                    # Use the grounded value from the first match
                    parameters[param] = entity_list[0]['grounded']
                elif param == 'gw':
                    # Gameweek should be an integer
                    parameters[param] = entity_list[0]['grounded']
                elif param == 'limit':
                    # Extract number from statistics if present
                    # Default to 10 if not specified
                    limit_value = 10
                    if entities.get('statistics'):
                        for stat in entities['statistics']:
                            # Try to extract number from phrases like "top 5", "best 10"
                            num_match = re.search(r'\d+', stat)
                            if num_match:
                                limit_value = int(num_match.group())
                                break
                    parameters[param] = limit_value
    
    # Check for missing required parameters (non-OPTIONAL matches)
    # This is a simple heuristic - you may want to make this more sophisticated
    missing_params = []
    for param in required_params:
        if param not in parameters:
            # Check if the parameter is in an OPTIONAL MATCH clause
            # If not, it's required
            optional_pattern = rf'OPTIONAL\s+MATCH.*\${param}\b'
            if not re.search(optional_pattern, query, re.IGNORECASE | re.DOTALL):
                missing_params.append(param)
    
    if missing_params:
        print(f"Warning: Missing required parameters: {missing_params}")
        print(f"Available entities: {list(entities.keys())}")
        return None, None
    
    return parameters


# ===== Cell 8 =====
config = {}
with open("config.txt", "r") as f:
    for line in f:
        key, value = line.strip().split("=", 1)
        config[key] = value

with driver.session() as session:
    result = session.run(
        """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
        OPTIONAL MATCH (p)-[:PLAYS_FOR]->(t:Team)
        RETURN
            p.code AS code,
            p.player_name    AS name,
            coalesce(pos.name, '') AS position,
            coalesce(head(collect(DISTINCT t.name)), '') AS team,
            SUM(r.goals_scored)      AS sum_goals,
            SUM(r.assists)           AS sum_assists,
            SUM(r.total_points)      AS sum_points,
            SUM(r.minutes)           AS sum_minutes,
            SUM(r.clean_sheets)      AS sum_clean_sheets,
            SUM(r.goals_conceded)    AS sum_goals_conceded,
            SUM(r.bps)               AS sum_bps,
            AVG(r.influence)         AS avg_influence,
            AVG(r.creativity)        AS avg_creativity,
            AVG(r.threat)            AS avg_threat,
            AVG(r.ict_index)         AS avg_ict_index
        """
    )

    df = pd.DataFrame(result.data())

print("Rows fetched from Neo4j:", len(df))


# ===== Cell 9 =====
def get_similar_players(player_name, mode="text", top_k=5):
    if mode == "text":
        embedding_prop = "embedding_text_l6_v2"
        index_name = "playerEmbeddingTextIndexL6V2"
    elif mode == "numeric":
        embedding_prop = "embedding_numeric"
        index_name = "playerEmbeddingNumericIndex"
    elif mode == "text_v2":
        embedding_prop = "embedding_text_mpnet_v2"
        index_name = "playerEmbeddingTextIndexMpnetV2"
    else:
        raise ValueError("mode must be 'text', 'text_v2', or 'numeric'")

    query = f"""
    MATCH (p:Player {{player_name: $name}})
    WITH p.{embedding_prop} AS query_vec
    CALL db.index.vector.queryNodes('{index_name}', $top_k, query_vec)
    YIELD node, score
    RETURN node.player_name AS similar_player, score
    ORDER BY score DESC
    """
    with driver.session() as session:
        results = session.run(query, name=player_name, top_k=top_k)
        return [(r["similar_player"], r["score"]) for r in results]



# ===== Cell 10 =====
def normalize_baseline_result(result_list):
    """
    Converts Neo4j baseline output into a unified format.
    Handles nodes (Player, Team, Position, Fixture), relationships, and aggregated stats.
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
                if isinstance(value, dict):  # from Neo4j node
                    item["players"].append({
                        "player_name": value.get("player_name", "Unknown"),
                        "player_code": value.get("code"),
                        "other_props": {k: v for k, v in value.items() if k not in ["player_name", "code"]}
                    })
                else:  # aggregated name string
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
                        "kickoff_time": value.get("kickoff_time"),
                        "season": value.get("season"),
                        "other_props": {k: v for k, v in value.items() if k not in ["fixture_number", "kickoff_time", "season"]}
                    })
                else:
                    item["fixtures"].append({"fixture": value})

            # Relationships (tuple format)
            elif key == "r" and isinstance(value, tuple) and len(value) == 3:
                start, rel, end = value
                item["relationships"].append({
                    "type": rel if isinstance(rel, str) else getattr(rel, "type", "unknown"),
                    "from": getattr(start, "get", lambda x, d=None: d)("player_name", getattr(start, "name", "Unknown")),
                    "to": getattr(end, "get", lambda x, d=None: d)("player_name", getattr(end, "name", "Unknown")),
                    "props": rel.items() if hasattr(rel, "items") else {}
                })

            # Aggregated stats
            elif key in ["assists", "minutes", "avg_form", "total_points", "goals", "points", "player1_points", "player2_points"]:
                item["aggregated_stats"][key] = value

            # Gameweek / Season nodes
            elif key == "g":
                if isinstance(value, dict):
                    item["gameweek"] = value
                else:
                    item["gameweek"] = {"value": value}

            # Player comparison
            elif key in ["player1", "player2"]:
                item["players"].append({"player_name": value})

            else:
                # fallback: store in aggregated_stats
                item["aggregated_stats"][key] = value

        unified.append(item)

    return unified


# ===== Cell 12 =====
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv(override=True)

# Verify keys are loaded
print("OpenAI Key loaded:", bool(os.getenv("OPENAI_KEY")))
print("OpenRouter Key loaded:", bool(os.getenv("OPEN_ROUTER_KEY")))
print("Google API Key loaded:", bool(os.getenv("GEMINI_KEY")))

OPENAI_KEY = KEY = os.getenv("OPENAI_KEY")
OPENROUTER_KEY = KEY = os.getenv("OPEN_ROUTER_KEY")
GEMINI_KEY = KEY = os.getenv("GEMINI_KEY")


# ===== Cell 13 =====
# Persona: Define the assistant's role with comprehensive guidelines
SystemPrompt = """You are an expert Fantasy Premier League (FPL) assistant with deep knowledge of player statistics, team performance, fixtures, and FPL strategy. You provide accurate, data-driven insights to help FPL managers make informed decisions.

Guidelines for your responses:
- Carefully analyze the context provided to extract relevant information
- Answer questions accurately using ONLY the information available in the context
- If the context doesn't contain enough information to answer the question, clearly state that
- Provide specific statistics, player names, team names, and gameweek data when available in the context
- Be concise but comprehensive in your answers
- Use FPL terminology correctly (e.g., GW for gameweek, xG for expected goals, xA for expected assists, ICT index, bonus points, BPS, etc.)
- When discussing player performance, include relevant metrics like points scored, goals, assists, clean sheets, and bonus points if available
- For team-related questions, reference fixtures, form, and statistics from the context
- Maintain an enthusiastic and knowledgeable tone about Fantasy Premier League
- Base all recommendations and insights strictly on the provided context to avoid hallucinations"""


# ===== Cell 14 =====
import time

def openai_generate(context: str, question: str) -> dict:
    """
    Generate response using OpenAI model.
    
    Args:
        context: The retrieved knowledge graph information (nodes, relationships, data)
        question: The user's question to answer
        
    Returns:
        Dictionary containing:
        - response: String response from the model
        - metrics: Dictionary with response_time, token_usage, and cost
    """
    openai_llm = ChatOpenAI(
        model="gpt-5.1",
        temperature=0.7,
        api_key=OPENAI_KEY,
        max_tokens=1000
    )
    
    # Structure: Persona (SystemMessage) + Context + Task (HumanMessage)
    messages = [
        SystemMessage(content=SystemPrompt),
        HumanMessage(content=f"""Context:
{context}

Task:
Answer the following question using ONLY the information provided in the context above. If the context doesn't contain enough information to answer the question, clearly state that. Be specific and cite relevant statistics, player names, or data from the context.

Question: {question}""")
    ]
    
    # Measure response time
    start_time = time.time()
    response = openai_llm.invoke(messages)
    end_time = time.time()
    
    # Extract token usage
    prompt_tokens = response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)
    completion_tokens = response.response_metadata.get('token_usage', {}).get('completion_tokens', 0)
    total_tokens = response.response_metadata.get('token_usage', {}).get('total_tokens', 0)
    
    # Calculate cost (GPT-5.1 pricing: $1.25 per 1M prompt tokens, $10 per 1M completion tokens)
    cost = (prompt_tokens / 1000000 * 1.25) + (completion_tokens / 1000000 * 10)
    
    return {
        "response": response.content,
        "metrics": {
            "response_time": round(end_time - start_time, 2),
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "cost": round(cost, 6)
        }
    }


# ===== Cell 15 =====
def openrouter_generate(context: str, question: str) -> dict:
    """
    Generate response using OpenRouter model.
    
    Args:
        context: The retrieved knowledge graph information (nodes, relationships, data)
        question: The user's question to answer
        
    Returns:
        Dictionary containing:
        - response: String response from the model
        - metrics: Dictionary with response_time, token_usage, and cost
    """
    openrouter_llm = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        temperature=0.7,
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=1000,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "LangChain Template"
        }
    )
    
    # Structure: Persona (SystemMessage) + Context + Task (HumanMessage)
    messages = [
        SystemMessage(content=SystemPrompt),
        HumanMessage(content=f"""Context:
{context}

Task:
Answer the following question using ONLY the information provided in the context above. If the context doesn't contain enough information to answer the question, clearly state that. Be specific and cite relevant statistics, player names, or data from the context.

Question: {question}""")
    ]
    
    # Measure response time
    start_time = time.time()
    response = openrouter_llm.invoke(messages)
    end_time = time.time()
    
    # Extract token usage
    prompt_tokens = response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)
    completion_tokens = response.response_metadata.get('token_usage', {}).get('completion_tokens', 0)
    total_tokens = response.response_metadata.get('token_usage', {}).get('total_tokens', 0)
    
    # Cost for free model is $0
    cost = 0.0
    
    return {
        "response": response.content,
        "metrics": {
            "response_time": round(end_time - start_time, 2),
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "cost": round(cost, 6)
        }
    }


# ===== Cell 16 =====
def gemini_generate(context: str, question: str) -> dict:
    """
    Generate response using Gemini model.
    
    Args:
        context: The retrieved knowledge graph information (nodes, relationships, data)
        question: The user's question to answer
        
    Returns:
        Dictionary containing:
        - response: String response from the model
        - metrics: Dictionary with response_time, token_usage, and cost
    """
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=GEMINI_KEY,
        max_tokens=1000
    )
    
    # Structure: Persona (SystemMessage) + Context + Task (HumanMessage)
    messages = [
        SystemMessage(content=SystemPrompt),
        HumanMessage(content=f"""Context:
{context}

Task:
Answer the following question using ONLY the information provided in the context above. If the context doesn't contain enough information to answer the question, clearly state that. Be specific and cite relevant statistics, player names, or data from the context.

Question: {question}""")
    ]
    
    # Measure response time
    start_time = time.time()
    response = gemini_llm.invoke(messages)
    end_time = time.time()
    
    # Extract token usage
    usage_metadata = response.usage_metadata
    prompt_tokens = usage_metadata.get('input_tokens', 0)
    completion_tokens = usage_metadata.get('output_tokens', 0)
    total_tokens = usage_metadata.get('total_tokens', 0)
    
    # Calculate cost (Gemini 2.5 Flash pricing: $0.03 per 1M input tokens, $2.50 per 1M output tokens)
    cost = (prompt_tokens / 1000000 * 0.03) + (completion_tokens / 1000000 * 2.50)
    
    return {
        "response": response.content,
        "metrics": {
            "response_time": round(end_time - start_time, 2),
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "cost": round(cost, 6)
        }
    }


# ===== Cell 17 =====
# =====================================================
# ENHANCED get_context() - Returns Cypher query too
# =====================================================

def get_context_enhanced(question, retrieval_method="both"):
    """
    Enhanced version that returns Cypher query and supports different retrieval methods.
    
    Args:
        question: User's question
        retrieval_method: "baseline", "embeddings", or "both"
    
    Returns:
        tuple: (unified_context, normalized_baseline, cypher_query, query_params)
    """
    # 1️⃣ Extract structured query info
    category, query = classify_intent(question)
    _, grounded_entities = extract_entities(question)
    query_params = populate_query(query, grounded_entities)
    
    # 2️⃣ Run baseline Cypher query (if selected)
    normalized_baseline = []
    cypher_query = query if query else "No query generated"
    
    if retrieval_method in ["baseline", "both"] and query_params is not None and query is not None:
        with driver.session() as session:
            result1 = session.run(query, query_params)
            baseline = result1.data()
        normalized_baseline = normalize_baseline_result(baseline)
    
    # 3️⃣ Prepare player names for embedding-based retrieval (if selected)
    embeddings_context = []
    if retrieval_method in ["embeddings", "both"]:
        players = grounded_entities.get("players", [])
        players_name = [player['grounded'] for player in players]
        
        # 4️⃣ Get embedding-based contexts
        for player_name in players_name:
            context = get_similar_players(player_name, mode="text_v2")
            embeddings_context.extend(context)
    
    # 5️⃣ Combine results
    unified_context = []
    
    # Add embedding-based results
    for player_name, score in embeddings_context:
        unified_context.append({
            "type": "semantic",
            "player_name": player_name,
            "similarity_score": score
        })
    
    unified_context.extend(normalized_baseline)
    
    return unified_context, normalized_baseline, cypher_query, query_params


# ===== Cell 18 =====
# =====================================================
# GRAPH VISUALIZATION FUNCTION
# =====================================================

import networkx as nx
import plotly.graph_objects as go

def create_knowledge_graph_visualization(baseline_context):
    """
    Create an interactive graph visualization from baseline context.
    
    Args:
        baseline_context: Normalized baseline results
    
    Returns:
        plotly figure object
    """
    G = nx.Graph()
    
    # Extract nodes and edges from baseline context
    for record in baseline_context:
        # Add player nodes
        for player in record.get("players", []):
            player_name = player.get("player_name", "Unknown")
            G.add_node(player_name, node_type="player", color="#3498db")
        
        # Add team nodes
        for team in record.get("teams", []):
            team_name = team.get("team_name", "Unknown")
            G.add_node(team_name, node_type="team", color="#e74c3c")
        
        # Add position nodes
        for position in record.get("positions", []):
            pos_name = position.get("position_name", "Unknown")
            G.add_node(pos_name, node_type="position", color="#2ecc71")
        
        # Add relationships as edges
        for rel in record.get("relationships", []):
            from_node = rel.get("from", "")
            to_node = rel.get("to", "")
            rel_type = rel.get("type", "")
            if from_node and to_node:
                G.add_edge(from_node, to_node, relationship=rel_type)
        
        # Create edges from aggregated stats (player connections)
        players = [p.get("player_name") for p in record.get("players", [])]
        if len(players) >= 2:
            # Connect compared players
            G.add_edge(players[0], players[1], relationship="compared_with")
    
    if len(G.nodes()) == 0:
        # Return empty figure if no nodes
        fig = go.Figure()
        fig.add_annotation(
            text="No graph data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )
        return fig
    
    # Create layout using spring layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Create edge traces
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                showlegend=False
            )
        )
    
    # Create node traces grouped by type
    node_traces = {}
    for node in G.nodes():
        node_type = G.nodes[node].get('node_type', 'unknown')
        color = G.nodes[node].get('color', '#999')
        
        if node_type not in node_traces:
            node_traces[node_type] = {
                'x': [], 'y': [], 'text': [],
                'color': color, 'name': node_type.capitalize()
            }
        
        x, y = pos[node]
        node_traces[node_type]['x'].append(x)
        node_traces[node_type]['y'].append(y)
        node_traces[node_type]['text'].append(node)
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    for trace in edge_trace:
        fig.add_trace(trace)
    
    # Add nodes
    for node_type, data in node_traces.items():
        fig.add_trace(go.Scatter(
            x=data['x'],
            y=data['y'],
            mode='markers+text',
            name=data['name'],
            text=data['text'],
            textposition="top center",
            marker=dict(
                size=20,
                color=data['color'],
                line=dict(width=2, color='white')
            ),
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title="Knowledge Graph Visualization",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        plot_bgcolor='rgba(250,250,250,0.9)'
    )
    
    return fig




# =====================
# ENHANCED STREAMLIT UI
# =====================
st.title("⚽ Fantasy Premier League Graph-RAG Assistant")
st.markdown("Ask questions about players, teams, fixtures, and more!")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    # Retrieval method selection
    retrieval_method = st.selectbox(
        "Retrieval Method",
        ["both", "baseline", "embeddings"],
        help="Choose how to retrieve context from the knowledge graph"
    )

    # Model comparison mode
    compare_models = st.checkbox(
        "Compare All Models",
        value=False,
        help="Run all 3 models simultaneously for comparison"
    )

    if not compare_models:
        selected_model = st.selectbox(
            "Select Model",
            ['Gemini 2.5 Flash', 'LLama 3.3 (Openrouter)', 'OpenAI GPT 5.1']
        )

    st.divider()

    st.header("ℹ️ About")
    st.markdown("""
    **Retrieval Methods:**
    - **Baseline**: Cypher queries only
    - **Embeddings**: Semantic similarity only
    - **Both**: Hybrid approach (recommended)

    **Features:**
    - Natural language queries
    - Knowledge graph retrieval
    - Multiple LLM options
    - Real-time model comparison
    - Graph visualization
    """)

    st.divider()

    st.header("📊 Database Info")
    if 'kg_entities' in globals():
        st.metric("Players", len(kg_entities['players']))
        st.metric("Teams", len(kg_entities['teams']))
        st.metric("Positions", len(kg_entities['positions']))
        st.metric("Seasons", len(kg_entities['seasons']))

# Main input
user_query = st.text_input(
    "Ask a question:",
    placeholder="e.g., Who performed better in GW15, Salah or Haaland?"
)

# Submit button
if st.button("Get Answer", type="primary"):
    if not user_query:
        st.error("Please enter a question!")
    else:
        with st.spinner("🔍 Processing your question..."):
            try:
                # Get context with enhanced function
                text_context, baseline_context, cypher_query, query_params = get_context_enhanced(
                    user_query,
                    retrieval_method=retrieval_method
                )

                # Display Cypher Query
                st.subheader("🔍 Cypher Query Executed")
                with st.expander("View Query Details", expanded=False):
                    st.code(cypher_query, language="cypher")
                    if query_params:
                        st.write("**Parameters:**")
                        st.json(query_params)

                # Display Graph Visualization
                st.subheader("📊 Knowledge Graph Visualization")
                if baseline_context:
                    fig = create_knowledge_graph_visualization(baseline_context)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No graph data to visualize")

                # Display retrieved data
                with st.expander("📊 Knowledge Graph Data Retrieved", expanded=False):
                    for i, record in enumerate(baseline_context, start=1):
                        st.markdown(f"**Result {i}**")

                        if record.get("players"):
                            st.markdown("**🧑 Players**")
                            st.json(record["players"])

                        if record.get("teams"):
                            st.markdown("**🏟 Teams**")
                            st.json(record["teams"])

                        if record.get("aggregated_stats"):
                            st.markdown("**📈 Aggregated Stats**")
                            st.json(record["aggregated_stats"])

                st.divider()

                # Model comparison or single model
                if compare_models:
                    st.subheader("🤖 Model Comparison")

                    # Run all models in parallel
                    import concurrent.futures

                    models = {
                        'Gemini 2.5 Flash': gemini_generate,
                        'LLama 3.3 (Openrouter)': openrouter_generate,
                        'OpenAI GPT 5.1': openai_generate
                    }

                    results = {}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        future_to_model = {
                            executor.submit(func, str(text_context), user_query): name
                            for name, func in models.items()
                        }

                        for future in concurrent.futures.as_completed(future_to_model):
                            model_name = future_to_model[future]
                            try:
                                results[model_name] = future.result()
                            except Exception as e:
                                st.error(f"Error with {model_name}: {str(e)}")

                    # Display results in columns
                    cols = st.columns(3)

                    for idx, (model_name, result) in enumerate(results.items()):
                        with cols[idx]:
                            st.markdown(f"### {model_name}")
                            st.info(result["response"])

                            st.markdown("**Metrics:**")
                            st.metric("Time", f"{result['metrics']['response_time']}s")
                            st.metric("Tokens", result['metrics']['token_usage']['total_tokens'])
                            st.metric("Cost", f"${result['metrics']['cost']}")

                    # Comparison table
                    st.subheader("📊 Metrics Comparison")
                    comparison_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'Response Time (s)': [r['metrics']['response_time'] for r in results.values()],
                        'Total Tokens': [r['metrics']['token_usage']['total_tokens'] for r in results.values()],
                        'Cost ($)': [r['metrics']['cost'] for r in results.values()]
                    })
                    st.dataframe(comparison_df, use_container_width=True)

                else:
                    # Single model mode
                    st.subheader("🤖 LLM Answer")

                    if selected_model == 'Gemini 2.5 Flash':
                        result = gemini_generate(str(text_context), user_query)
                    elif selected_model == 'LLama 3.3 (Openrouter)':
                        result = openrouter_generate(str(text_context), user_query)
                    else:
                        result = openai_generate(str(text_context), user_query)

                    st.info(result["response"])

                    st.subheader("📊 Metrics")
                    col1, col2, col3, col4, col5 = st.columns(5)

                    with col1:
                        st.metric("Response Time", f"{result['metrics']['response_time']}s")
                    with col2:
                        st.metric("Input Tokens", result['metrics']['token_usage']['prompt_tokens'])
                    with col3:
                        st.metric("Output Tokens", result['metrics']['token_usage']['completion_tokens'])
                    with col4:
                        st.metric("Total Tokens", result['metrics']['token_usage']['total_tokens'])
                    with col5:
                        st.metric("Cost", f"${result['metrics']['cost']}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)
