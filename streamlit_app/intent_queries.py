"""
Intent Queries Module
Defines intent categories and their corresponding Cypher query templates

Intent Classification helps route user queries to appropriate retrieval strategies.
Each intent maps to a specific Cypher query pattern for structured data retrieval.
"""

# Available intent categories for Fantasy Premier League queries
INTENTION_CATEGORIES = [
    "player_basic_info",
    "player_season_stats",
    "player_gw_stats",
    "fixture_details",
    "team_fixtures",
    "team_players",
    "top_players_position",
    "compare_players",
    "gameweek_summary",
    "player_vs_opponent"
]

# Mapping of intents to Cypher query templates
# Parameters are denoted with $ prefix (e.g., $player, $season)
INTENTION_MAP = {
    # Get basic information about a player (who they are, position, team)
    "player_basic_info": """
MATCH (p:Player {player_name: $player})
OPTIONAL MATCH (p)-[:PLAYS_AS]->(pos:Position)
OPTIONAL MATCH (p)-[:PLAYS_FOR {season: $season}]->(t:Team)
RETURN p, pos, t;""",

    # Get a player's overall performance stats for a season
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

    # Get a player's performance in a specific gameweek
    "player_gw_stats": """
MATCH (p:Player {player_name: $player})-[r:PLAYED_IN]->(f:Fixture)
MATCH (g:Gameweek {season: $season, GW_number: $gw})-[:HAS_FIXTURE]->(f)
RETURN p, r, f, properties(r) AS r_props;
""",

    # Get details about a specific fixture/match
    "fixture_details": """
MATCH (f:Fixture {season: $season, fixture_number: $fixture}) 
MATCH (f)-[:HAS_HOME_TEAM]->(home:Team) 
MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team) 
MATCH (p:Player)-[r:PLAYED_IN]->(f) 
RETURN f, home, away, p, r, properties(r) AS r_props;
""",

    # Get all fixtures for a specific team
    "team_fixtures": """
MATCH (t:Team {name: $team})
MATCH (f:Fixture)-[r:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)
RETURN f,r,t,properties(r) AS r_props ORDER BY f.kickoff_time;
""",

    # Get all players in a team for a season
    "team_players": """
MATCH (t:Team {name: $team})
MATCH (p:Player)-[:PLAYS_FOR {season: $season}]->(t)
RETURN p;
""",

    # Get top players by points in a specific position
    "top_players_position": """
MATCH (p:Player)-[:PLAYS_AS]->(:Position {name: $position})
MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek)<-[:HAS_GW]-(s:Season {season_name: $season})
RETURN p.player_name AS player, SUM(r.total_points) AS points
ORDER BY points DESC LIMIT $limit;
""",

    # Compare two players' overall performance
    "compare_players": """
MATCH (p1:Player {player_name: $player1})-[r1:PLAYED_IN]->(f1:Fixture)
MATCH (p2:Player {player_name: $player2})-[r2:PLAYED_IN]->(f2:Fixture)
RETURN p1.player_name AS player1, SUM(r1.total_points) AS p1_points,
       p2.player_name AS player2, SUM(r2.total_points) AS p2_points;
""",

    # Get summary of all fixtures and performance in a gameweek
    "gameweek_summary": """
MATCH (gameweek:Gameweek {season: $season, GW_number: $gw})-[:HAS_FIXTURE]->(f)
MATCH (p:Player)-[r:PLAYED_IN]->(f)
RETURN gameweek, f, p, r, properties(r) AS r_props
ORDER BY f.fixture_number;
""",

    # Get a player's performance against a specific opponent
    "player_vs_opponent": """
MATCH (p:Player {player_name: $player})-[r:PLAYED_IN]->(f:Fixture)
MATCH (p)-[:PLAYED_AGAINST]->(opp:Team {name: $opponent})
RETURN p.player_name AS player, opp.name AS opponent,
       SUM(r.goals_scored) AS goals,
       SUM(r.assists) AS assists,
       SUM(r.total_points) AS points;
"""
}

# Intent descriptions for better UI display
INTENT_DESCRIPTIONS = {
    "player_basic_info": "Asking who a player is, their position, or their team",
    "player_season_stats": "Asking about a player's overall seasonal performance",
    "player_gw_stats": "Asking about a player's performance in a specific gameweek",
    "fixture_details": "Asking about a specific match, its teams, or players in it",
    "team_fixtures": "Asking about a team's upcoming or past fixtures",
    "team_players": "Asking which players belong to a team",
    "top_players_position": "Asking for ranking or best players in a position/season",
    "compare_players": "Comparing two players statistically",
    "gameweek_summary": "Asking about all fixtures or events in a specific GW",
    "player_vs_opponent": "Asking how a player performed against a specific team"
}
