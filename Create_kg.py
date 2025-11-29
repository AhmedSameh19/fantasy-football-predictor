from neo4j import GraphDatabase
import pandas as pd

config = {}
with open("config.txt", "r") as f:
    for line in f:
        key, value = line.strip().split("=", 1)
        config[key] = value

driver = GraphDatabase.driver(config["URI"], auth=(config["USERNAME"], config["PASSWORD"]))


df = pd.read_csv("fpl_two_seasons.csv", header=0)

with driver.session() as session:
    batch_size = 1000

    data = []
    for row in df.itertuples():
        data.append({
            'season': row.season,
            'GW': row.GW,
            'fixture': row.fixture,
            'home_team': row.home_team,
            'away_team': row.away_team,
            'name': row.name,
            'element': row.element,
            'position': row.position,
            'kickoff': row.kickoff_time,
            'minutes': row.minutes,
            'goals_scored': row.goals_scored,
            'assists': row.assists,
            'total_points': row.total_points,
            'bonus': row.bonus,
            'clean_sheets': row.clean_sheets,
            'goals_conceded': row.goals_conceded,
            'own_goals': row.own_goals,
            'penalties_saved': row.penalties_saved,
            'penalties_missed': row.penalties_missed,
            'yellow_cards': row.yellow_cards,
            'red_cards': row.red_cards,
            'saves': row.saves,
            'bps': row.bps,
            'influence': row.influence,
            'creativity': row.creativity,
            'threat': row.threat,
            'ict_index': row.ict_index,
            'form': row.form
        })

    # Process in batches
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        session.run(
            """
            UNWIND $batch as row
            MERGE (s:Season {season_name: row.season})
            MERGE (g:Gameweek {season: row.season, GW_number: row.GW})
            MERGE (f:Fixture {season: row.season, fixture_number: row.fixture})
                SET f.kickoff_time = row.kickoff
            MERGE (t_home:Team {name: row.home_team})
            MERGE (t_away:Team {name: row.away_team})
            MERGE (p:Player {player_name: row.name, player_element: row.element})
            MERGE (pos:Position {name: row.position})

            MERGE (s)-[:HAS_GW]->(g)
            MERGE (g)-[:HAS_FIXTURE]->(f)
            MERGE (f)-[:HAS_HOME_TEAM]->(t_home)
            MERGE (f)-[:HAS_AWAY_TEAM]->(t_away)
            MERGE (p)-[:PLAYS_AS]->(pos)
            MERGE (p)-[r:PLAYED_IN]->(f)
            SET r.minutes = row.minutes,
                r.goals_scored = row.goals_scored,
                r.assists = row.assists,
                r.total_points = row.total_points,
                r.bonus = row.bonus,
                r.clean_sheets = row.clean_sheets,
                r.goals_conceded = row.goals_conceded,
                r.own_goals = row.own_goals,
                r.penalties_saved = row.penalties_saved,
                r.penalties_missed = row.penalties_missed,
                r.yellow_cards = row.yellow_cards,
                r.red_cards = row.red_cards,
                r.saves = row.saves,
                r.bps = row.bps,
                r.influence = row.influence,
                r.creativity = row.creativity,
                r.threat = row.threat,
                r.ict_index = row.ict_index,
                r.form = row.form
            """,
            batch=batch
        )
        print(f"Imported batch {i // batch_size + 1}: rows {i} to {min(i + batch_size, len(data))}")

    print("Data imported successfully.")

driver.close()