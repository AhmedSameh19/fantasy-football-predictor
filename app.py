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

# Load environment variables
load_dotenv(override=True)

# =====================
# Configuration
# =====================
config = {}
with open("config.txt", "r") as f:
    for line in f:
        key, value = line.strip().split("=", 1)
        config[key] = value

# API Keys
OPENAI_KEY = os.getenv("OPENAI_KEY")
OPENROUTER_KEY = os.getenv("OPEN_ROUTER_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")

# =====================
# Neo4j Setup
# =====================
neo4j_driver = GraphDatabase.driver(config["URI"], auth=(config["USERNAME"], config["PASSWORD"]))
driver = neo4j_driver  # For compatibility

# =====================
# Intent Categories and Queries
# =====================
intention_categories = ["player_basic_info", "player_season_stats", "player_gw_stats",
                        "fixture_details", "team_fixtures", "team_players", "top_players_position",
                        "compare_players", "gameweek_summary", "player_vs_opponent"]

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
    "player_gw_stats": """
MATCH (p:Player {player_name: $player})-[r:PLAYED_IN]->(f:Fixture)
MATCH (g:Gameweek {season: $season, GW_number: $gw})-[:HAS_FIXTURE]->(f)
RETURN p, r, f;
""",
    "fixture_details": """
MATCH (p:Player {player_name: $player})-[r:PLAYED_IN]->(f:Fixture)
MATCH (g:Gameweek {season: $season, GW_number: $gw})-[:HAS_FIXTURE]->(f)
RETURN p, r, f;
""",
    "team_fixtures": """
MATCH (t:Team {name: $team})
MATCH (f:Fixture)-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)
RETURN f ORDER BY f.kickoff_time;
""",
    "team_players": """
MATCH (t:Team {name: $team})
MATCH (p:Player)-[:PLAYS_FOR {season: $season}]->(t)
RETURN p;
""",
    "top_players_position": """
MATCH (p:Player)-[:PLAYS_AS]->(:Position {name: $position})
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek)<-[:HAS_GW]-(s:Season {season_name: $season})
RETURN p.player_name AS player, SUM(r.total_points) AS points
ORDER BY points DESC LIMIT $limit;
""",
    "compare_players": """
MATCH (p1:Player {player_name: $player1})-[r1:PLAYED_IN]->(f1:Fixture)
MATCH (p2:Player {player_name: $player2})-[r2:PLAYED_IN]->(f2:Fixture)
RETURN p1.player_name AS player1, SUM(r1.total_points) AS p1_points,
       p2.player_name AS player2, SUM(r2.total_points) AS p2_points;
""",
    "gameweek_summary": """
MATCH (g:Gameweek {season: $season, GW_number: $gw})-[:HAS_FIXTURE]->(f)
MATCH (p:Player)-[r:PLAYED_IN]->(f)
RETURN g, f, p, r
ORDER BY f.fixture_number;
""",
    "player_vs_opponent": """
MATCH (p:Player {player_name: $player})-[r:PLAYED_IN]->(f:Fixture)
MATCH (p)-[:PLAYED_AGAINST]->(opp:Team {name: $opponent})
RETURN p.player_name AS player, opp.name AS opponent,
       SUM(r.goals_scored) AS goals,
       SUM(r.assists) AS assists,
       SUM(r.total_points) AS points;
"""
}

# =====================
# OpenRouter Client
# =====================
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)

def call_open_router(prompt: str) -> str:
    completion = openrouter_client.chat.completions.create(
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
        max_tokens=1000
    )
    return completion.choices[0].message.content

# =====================
# Knowledge Graph Entity Caching
# =====================
@st.cache_data
def get_kg_entities():
    """Retrieve all entity values from the knowledge graph."""
    with neo4j_driver.session() as session:
        players = session.run("MATCH (p:Player) RETURN p.player_name as name").data()
        player_names = [p['name'] for p in players if p['name']]

        teams = session.run("MATCH (t:Team) RETURN t.name as name").data()
        team_names = [t['name'] for t in teams if t['name']]

        positions = session.run("MATCH (pos:Position) RETURN pos.name as name").data()
        position_names = [pos['name'] for pos in positions if pos['name']]

        seasons = session.run("MATCH (s:Season) RETURN s.season_name as name").data()
        season_names = [s['name'] for s in seasons if s['name']]

        gameweeks = session.run("MATCH (g:Gameweek) RETURN DISTINCT g.GW_number as gw ORDER BY gw").data()
        gw_numbers = [gw['gw'] for gw in gameweeks if gw['gw']]

    return {
        'players': player_names,
        'teams': team_names,
        'positions': position_names,
        'seasons': season_names,
        'gameweeks': gw_numbers
    }

kg_entities = get_kg_entities()

# =====================
# Helper Functions
# =====================
def classify_intent(user_input):
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

User Input: "Compare Son and Rashford this season."
Category: compare_players

Now classify the user's input below.
Return ONLY the category name from the list above.

User Input: "{user_input}"
Category:
"""
    category = call_open_router(prompt).strip()
    if category not in intention_categories:
        st.warning(f"LLM returned unexpected category: {category}")
        category = "Unknown"
    return category, intention_map.get(category)

def extract_entities(user_input):
    """Extract and ground entities from user input using the knowledge graph."""
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

    # Parse LLM response
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

    # Ground entities
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

    # Ground positions
    position_mapping = {
        'goalkeeper': 'GK', 'gk': 'GK',
        'defender': 'DEF', 'def': 'DEF',
        'midfielder': 'MID', 'mid': 'MID',
        'forward': 'FWD', 'fwd': 'FWD',
        'striker': 'FWD', 'attacker': 'FWD'
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

    # Ground seasons
    for season in extracted['seasons']:
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

    # Extract gameweek numbers
    for gw in extracted['gameweeks']:
        gw_match = re.search(r'\d+', gw)
        if gw_match:
            gw_num = int(gw_match.group())
            if gw_num in kg_entities['gameweeks']:
                grounded_entities['gameweeks'].append({
                    'original': gw,
                    'grounded': gw_num
                })

    grounded_entities['statistics'] = extracted['statistics']

    return extracted, grounded_entities

def populate_query(query, entities):
    """Populate a Cypher query template with extracted entities."""
    if query is None:
        return None

    param_pattern = r'\$(\w+)'
    required_params = set(re.findall(param_pattern, query))

    parameters = {}

    param_to_entity_map = {
        'player': 'players',
        'player1': 'players',
        'player2': 'players',
        'team': 'teams',
        'opponent': 'teams',
        'position': 'positions',
        'season': 'seasons',
        'gw': 'gameweeks',
        'limit': 'statistics'
    }

    for param in required_params:
        entity_type = param_to_entity_map.get(param)

        if entity_type and entity_type in entities:
            entity_list = entities[entity_type]

            if entity_list:
                if param == 'player1' and len(entity_list) >= 1:
                    parameters[param] = entity_list[0]['grounded']
                elif param == 'player2' and len(entity_list) >= 2:
                    parameters[param] = entity_list[1]['grounded']
                elif param in ['player', 'team', 'opponent', 'position', 'season']:
                    parameters[param] = entity_list[0]['grounded']
                elif param == 'gw':
                    parameters[param] = entity_list[0]['grounded']
                elif param == 'limit':
                    limit_value = 10
                    if entities.get('statistics'):
                        for stat in entities['statistics']:
                            num_match = re.search(r'\d+', stat)
                            if num_match:
                                limit_value = int(num_match.group())
                                break
                    parameters[param] = limit_value

    missing_params = []
    for param in required_params:
        if param not in parameters:
            optional_pattern = rf'OPTIONAL\s+MATCH.*\${param}\b'
            if not re.search(optional_pattern, query, re.IGNORECASE | re.DOTALL):
                missing_params.append(param)

    if missing_params:
        st.warning(f"Missing required parameters: {missing_params}")
        return None

    return parameters

def normalize_baseline_result(result_list):
    """Converts Neo4j baseline output into a unified format."""
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
            if key == "p" or key == "player" or key.startswith("player"):
                if isinstance(value, dict):
                    item["players"].append({
                        "player_name": value.get("player_name", "Unknown"),
                        "player_code": value.get("code"),
                        "other_props": {k: v for k, v in value.items() if k not in ["player_name", "code"]}
                    })
                else:
                    item["players"].append({"player_name": value})

            elif key == "t":
                item["teams"].append({
                    "team_name": value.get("name", "Unknown") if isinstance(value, dict) else str(value)
                })

            elif key == "pos":
                item["positions"].append({
                    "position_name": value.get("name", "Unknown") if isinstance(value, dict) else str(value)
                })

            elif key in ["assists", "minutes", "avg_form", "total_points", "goals", "points", "p1_points", "p2_points"]:
                item["aggregated_stats"][key] = value

            elif key in ["player1", "player2"]:
                item["players"].append({"player_name": value})

            else:
                item["aggregated_stats"][key] = value

        unified.append(item)

    return unified

def get_context(question):
    """Main function to retrieve context from knowledge graph."""
    category, query = classify_intent(question)
    _, grounded_entities = extract_entities(question)
    query_params = populate_query(query, grounded_entities)

    normalized_baseline = []
    if query_params is not None and query is not None:
        with neo4j_driver.session() as session:
            result1 = session.run(query, query_params)
            baseline = result1.data()
        normalized_baseline = normalize_baseline_result(baseline)

    unified_context = []
    unified_context.extend(normalized_baseline)

    return unified_context, normalized_baseline

# =====================
# LLM Generation Functions
# =====================
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

def openai_generate(context: str, question: str) -> dict:
    openai_llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7,
        api_key=OPENAI_KEY,
        max_tokens=1000
    )

    messages = [
        SystemMessage(content=SystemPrompt),
        HumanMessage(content=f"""Context:
{context}

Task:
Answer the following question using ONLY the information provided in the context above. If the context doesn't contain enough information to answer the question, clearly state that. Be specific and cite relevant statistics, player names, or data from the context.

Question: {question}""")
    ]

    start_time = time.time()
    response = openai_llm.invoke(messages)
    end_time = time.time()

    prompt_tokens = response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)
    completion_tokens = response.response_metadata.get('token_usage', {}).get('completion_tokens', 0)
    total_tokens = response.response_metadata.get('token_usage', {}).get('total_tokens', 0)

    cost = (prompt_tokens / 1000000 * 2.5) + (completion_tokens / 1000000 * 10)

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

def openrouter_generate(context: str, question: str) -> dict:
    openrouter_llm = ChatOpenAI(
        model="meta-llama/llama-3.3-70b-instruct:free",
        temperature=0.7,
        api_key=OPENROUTER_KEY,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=1000,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "FPL Assistant"
        }
    )

    messages = [
        SystemMessage(content=SystemPrompt),
        HumanMessage(content=f"""Context:
{context}

Task:
Answer the following question using ONLY the information provided in the context above. If the context doesn't contain enough information to answer the question, clearly state that. Be specific and cite relevant statistics, player names, or data from the context.

Question: {question}""")
    ]

    start_time = time.time()
    response = openrouter_llm.invoke(messages)
    end_time = time.time()

    prompt_tokens = response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)
    completion_tokens = response.response_metadata.get('token_usage', {}).get('completion_tokens', 0)
    total_tokens = response.response_metadata.get('token_usage', {}).get('total_tokens', 0)

    return {
        "response": response.content,
        "metrics": {
            "response_time": round(end_time - start_time, 2),
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            "cost": 0.0
        }
    }

def gemini_generate(context: str, question: str) -> dict:
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.7,
        google_api_key=GEMINI_KEY,
        max_tokens=1000
    )

    messages = [
        SystemMessage(content=SystemPrompt),
        HumanMessage(content=f"""Context:
{context}

Task:
Answer the following question using ONLY the information provided in the context above. If the context doesn't contain enough information to answer the question, clearly state that. Be specific and cite relevant statistics, player names, or data from the context.

Question: {question}""")
    ]

    start_time = time.time()
    response = gemini_llm.invoke(messages)
    end_time = time.time()

    usage_metadata = response.usage_metadata
    prompt_tokens = usage_metadata.get('input_tokens', 0)
    completion_tokens = usage_metadata.get('output_tokens', 0)
    total_tokens = usage_metadata.get('total_tokens', 0)

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

# =====================
# Streamlit UI
# =====================
st.set_page_config(
    page_title="Fantasy Premier League Assistant",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Fantasy Premier League Graph-RAG Assistant")
st.markdown("Ask questions about players, teams, fixtures, and more!")

# User input
user_query = st.text_input("Ask a question:", placeholder="e.g., Who performed better in GW15, Salah or Haaland?")

# Model selection
models = ['Gemini 2.0 Flash', 'LLama 3.3 (Openrouter)', 'OpenAI GPT-4o']
selected_model = st.selectbox("Select a model", models)

# Submit button
if st.button("Get Answer", type="primary"):
    if not user_query:
        st.error("Please enter a question!")
    else:
        with st.spinner("🔍 Processing your question..."):
            try:
                # Get context
                text_context, baseline_context = get_context(user_query)

                # Display knowledge graph data
                st.subheader("📊 Knowledge Graph Data Retrieved:")

                for i, record in enumerate(baseline_context, start=1):
                    with st.expander(f"Result {i}", expanded=True):

                        if record["players"]:
                            st.markdown("### 🧑 Players")
                            st.json(record["players"])

                        if record["teams"]:
                            st.markdown("### 🏟 Teams")
                            st.json(record["teams"])

                        if record["positions"]:
                            st.markdown("### 📍 Positions")
                            st.json(record["positions"])

                        if record["fixtures"]:
                            st.markdown("### 📅 Fixtures")
                            st.json(record["fixtures"])

                        if record["relationships"]:
                            st.markdown("### 🔗 Relationships")
                            st.json(record["relationships"])

                        if record["aggregated_stats"]:
                            st.markdown("### 📈 Aggregated Stats")
                            st.json(record["aggregated_stats"])

                st.divider()

                # Generate LLM answer
                st.subheader("🤖 LLM Answer:")

                if selected_model == 'Gemini 2.0 Flash':
                    result = gemini_generate(str(text_context), user_query)
                elif selected_model == 'LLama 3.3 (Openrouter)':
                    result = openrouter_generate(str(text_context), user_query)
                else:
                    result = openai_generate(str(text_context), user_query)

                # Display the answer
                st.info(result["response"])

                # Display metrics
                st.subheader("📊 Metrics:")
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

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app uses Graph-RAG (Retrieval-Augmented Generation) to answer questions about Fantasy Premier League.

    **Features:**
    - Natural language queries
    - Knowledge graph retrieval
    - Multiple LLM options
    - Detailed metrics

    **Example queries:**
    - "Who performed better in GW15, Salah or Haaland?"
    - "Show me the top 5 midfielders"
    - "What are Arsenal's upcoming fixtures?"
    """)

    st.divider()

    st.header("📊 Database Info")
    st.metric("Players", len(kg_entities['players']))
    st.metric("Teams", len(kg_entities['teams']))
    st.metric("Positions", len(kg_entities['positions']))
    st.metric("Seasons", len(kg_entities['seasons']))