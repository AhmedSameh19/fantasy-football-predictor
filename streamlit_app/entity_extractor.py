"""
Entity Extractor Module
Extracts and grounds entities from user queries using OpenAI and knowledge graph

Entity extraction identifies relevant entities (players, teams, etc.) from user input using OpenAI.
Entity grounding validates and maps extracted entities to actual values in the knowledge graph
using fuzzy string matching.
"""

import re
import json
from difflib import get_close_matches
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config import OPENAI_KEY
from database import get_kg_entities


def call_llm_for_entity_extraction(user_input: str) -> dict:
    """
    Call OpenAI API for entity extraction with structured JSON output.

    Uses GPT-5-mini with JSON mode to extract entities in a structured format,
    eliminating the need for manual parsing and ensuring consistent output.

    Args:
        user_input: The user's natural language query

    Returns:
        dict: Structured entities with keys:
            - players: List of player names
            - teams: List of team names
            - positions: List of positions
            - seasons: List of season references
            - gameweeks: List of gameweek references
            - statistics: List of statistical metrics
            - time_references: List of temporal references
    """
    openai_llm = ChatOpenAI(
        model="gpt-5-mini-2025-08-07",
        temperature=0.1,
        api_key=OPENAI_KEY,
        max_tokens=10000,
        model_kwargs={"response_format": {"type": "json_object"}}  # Enforce JSON output
    )

    prompt = f"""You are an intelligent entity extraction system for Fantasy Premier League (FPL) queries.

CONTEXT: Fantasy Premier League is a fantasy football game where users manage virtual teams of real Premier League players. You need to extract entities from user questions about:
- Player statistics (goals, assists, points, clean sheets, bonus points, etc.)
- Team performance and fixtures
- Gameweek-specific data
- Season-long performance comparisons
- Player positions and rankings

YOUR TASK:
Extract ALL relevant entities from the user's query and return them in a structured JSON format. Be smart about variations and synonyms.

ENTITY TYPES TO EXTRACT:

1. **players**: Player names or references
   - Include: Full names, last names, nicknames
   - Examples: "Salah", "Mohamed Salah", "Haaland", "Son", "Trent Alexander-Arnold"

2. **teams**: Premier League team names
   - Include: Full names, common abbreviations, nicknames
   - Examples: "Arsenal", "Liverpool", "Man City", "Manchester City", "Spurs", "Tottenham"

3. **positions**: Player positions (IMPORTANT: Recognize all variations)
   - Goalkeeper variations: "goalkeeper", "goalkeepers", "keeper", "keepers", "GK"
   - Defender variations: "defender", "defenders", "defence", "def", "DEF"
   - Midfielder variations: "midfielder", "midfielders", "midfield", "mid", "MID"
   - Forward variations: "forward", "forwards", "striker", "strikers", "attacker", "attackers", "FWD"
   - Extract the SINGULAR form only: "goalkeeper", "defender", "midfielder", "forward"

4. **seasons**: Season references
   - Include: Explicit seasons, relative references
   - Examples: "2023/24", "2020/21", "this season", "current season", "last season", "previous season"

5. **gameweeks**: Gameweek numbers or references
   - Include: GW numbers, gameweek mentions
   - Examples: "GW 5", "gameweek 10", "15", "GW15", "week 5"
   - Extract just the number: "5", "10", "15"

6. **fixtures**: Fixture numbers or match references
   - Include: Fixture numbers, match IDs
   - Examples: "Fixture 12", "fixture 162", "match 45"
   - Extract just the number: "12", "162", "45"

7. **statistics**: Performance metrics and statistical measures
   - Include: Goals, assists, points, clean sheets, bonus, saves, xG, xA, ICT, BPS, etc.
   - Examples: "goals", "assists", "points", "FPL points", "clean sheets", "top scoring", "most assists"

8. **time_references**: Any temporal context
   - Include: Season references, gameweek references, match references
   - Examples: "this season", "gameweek 5", "last match", "GW 10", "2023/24"

IMPORTANT RULES:
- Extract positions in SINGULAR form even if user uses plural (e.g., "goalkeepers" → "goalkeeper")
- Recognize synonyms (e.g., "attacker" = "forward", "keeper" = "goalkeeper")
- For positions, extract: "goalkeeper", "defender", "midfielder", or "forward" ONLY
- Extract ALL players mentioned, even if multiple
- Extract team names even if abbreviated or nicknamed
- Be comprehensive - extract all relevant entities

EXAMPLES:

User Input: "Who is the top scoring midfielder this season?"
Output: {{
  "players": [],
  "teams": [],
  "positions": ["midfielder"],
  "seasons": ["this season"],
  "gameweeks": [],
  "fixtures": [],
  "statistics": ["top scoring"],
  "time_references": ["this season"]
}}

User Input: "How many goals did Salah score in gameweek 5?"
Output: {{
  "players": ["Salah"],
  "teams": [],
  "positions": [],
  "seasons": [],
  "gameweeks": ["5"],
  "fixtures": [],
  "statistics": ["goals"],
  "time_references": ["gameweek 5"]
}}

User Input: "Compare Son and Rashford this season"
Output: {{
  "players": ["Son", "Rashford"],
  "teams": [],
  "positions": [],
  "seasons": ["this season"],
  "gameweeks": [],
  "fixtures": [],
  "statistics": [],
  "time_references": ["this season"]
}}

User Input: "Who plays for Arsenal this season?"
Output: {{
  "players": [],
  "teams": ["Arsenal"],
  "positions": [],
  "seasons": ["this season"],
  "gameweeks": [],
  "fixtures": [],
  "statistics": [],
  "time_references": ["this season"]
}}

User Input: "Show me the best goalkeepers in the 2020/21 season"
Output: {{
  "players": [],
  "teams": [],
  "positions": ["goalkeeper"],
  "seasons": ["2020/21"],
  "gameweeks": [],
  "fixtures": [],
  "statistics": ["best"],
  "time_references": ["2020/21"]
}}

User Input: "Which attackers scored the most goals for Liverpool in GW 15?"
Output: {{
  "players": [],
  "teams": ["Liverpool"],
  "positions": ["forward"],
  "seasons": [],
  "gameweeks": ["15"],
  "fixtures": [],
  "statistics": ["most goals"],
  "time_references": ["GW 15"]
}}

User Input: "Top 10 defenders by FPL points this season"
Output: {{
  "players": [],
  "teams": [],
  "positions": ["defender"],
  "seasons": ["this season"],
  "gameweeks": [],
  "fixtures": [],
  "statistics": ["top 10", "FPL points"],
  "time_references": ["this season"]
}}

User Input: "How many assists did Man City midfielders get in gameweek 7?"
Output: {{
  "players": [],
  "teams": ["Man City"],
  "positions": ["midfielder"],
  "seasons": [],
  "gameweeks": ["7"],
  "fixtures": [],
  "statistics": ["assists"],
  "time_references": ["gameweek 7"]
}}

User Input: "How did Salah perform in fixture 162?"
Output: {{
  "players": ["Salah"],
  "teams": [],
  "positions": [],
  "seasons": [],
  "gameweeks": [],
  "fixtures": ["162"],
  "statistics": ["perform"],
  "time_references": ["fixture 162"]
}}

Now extract entities from the following Fantasy Premier League query:

User Input: "{user_input}"
"""

    messages = [HumanMessage(content=prompt)]
    response = openai_llm.invoke(messages)
    print(f"Entity Extraction Response: {response.content}")

    # Parse JSON response
    try:
        entities = json.loads(response.content)

        # Ensure all required keys exist with default empty lists
        required_keys = ['players', 'teams', 'positions', 'seasons', 'gameweeks', 'fixtures', 'statistics', 'time_references']
        for key in required_keys:
            if key not in entities:
                entities[key] = []
            # Ensure values are lists
            if not isinstance(entities[key], list):
                entities[key] = [entities[key]] if entities[key] else []

        return entities
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON response: {response.content}")
        # Return empty structure on error
        return {
            'players': [],
            'teams': [],
            'positions': [],
            'seasons': [],
            'gameweeks': [],
            'fixtures': [],
            'statistics': [],
            'time_references': []
        }


def extract_entities(user_input: str):
    """
    Extract and ground entities from user input using OpenAI and knowledge graph.

    Process:
    1. Use OpenAI GPT-4o-mini with JSON mode to extract entities in structured format
    2. Ground entities against knowledge graph using fuzzy string matching
    3. Return both raw extractions and grounded (validated) entities

    Args:
        user_input: The user's natural language query

    Returns:
        tuple: (extracted_entities, grounded_entities)
            - extracted_entities: Raw entities from OpenAI (structured JSON)
            - grounded_entities: Validated entities matched to KG
    """
    # Get cached knowledge graph entities for grounding
    kg_entities = get_kg_entities()

    # Step 1: Use OpenAI with structured JSON output to extract entities
    extracted = call_llm_for_entity_extraction(user_input)

    # Step 3: Ground entities against the knowledge graph
    grounded_entities = {
        'players': [],
        'teams': [],
        'positions': [],
        'seasons': [],
        'gameweeks': [],
        'fixtures': [],
        'statistics': [],
        'time_references': extracted['time_references']
    }

    # Ground players (fuzzy string matching)
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

    # Ground positions (normalize to KG format: GK, DEF, MID, FWD)
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

    # Ground seasons (handle relative references like "this season", "last season")
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
        else:
            matches = get_close_matches(season, kg_entities['seasons'], n=1, cutoff=0.2)
            if matches:
                grounded_entities['seasons'].append({
                    'original': season,
                    'grounded': matches[0],
                    'is_relative': False
                })

    # Extract gameweek numbers (extract numeric value)
    for gw in extracted['gameweeks']:
        gw_match = re.search(r'\d+', gw)
        if gw_match:
            gw_num = int(gw_match.group())
            if gw_num in kg_entities['gameweeks']:
                grounded_entities['gameweeks'].append({
                    'original': gw,
                    'grounded': gw_num
                })

    # Extract fixture numbers (extract numeric value)
    for fixture in extracted['fixtures']:
        fixture_match = re.search(r'\d+', fixture)
        if fixture_match:
            fixture_num = int(fixture_match.group())
            grounded_entities['fixtures'].append({
                'original': fixture,
                'grounded': fixture_num
            })

    # Keep statistics as-is (performance metrics like "goals", "points", etc.)
    grounded_entities['statistics'] = extracted['statistics']

    return extracted, grounded_entities
