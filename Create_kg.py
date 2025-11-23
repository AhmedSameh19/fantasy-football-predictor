from neo4j import GraphDatabase
import pandas as pd

URI = "bolt://localhost:7687"
driver = GraphDatabase.driver(URI)

df = pd.read_csv("fpl_two_seasons.csv", header=0)

with driver.session() as session:

    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Season) REQUIRE s.season_name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gameweek) REQUIRE (g.season, g.GW_number) IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:Fixture) REQUIRE (f.season, f.fixture_number) IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Team) REQUIRE t.name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Player) REQUIRE (p.player_name, p.player_element) IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (pos:Position) REQUIRE pos.name IS UNIQUE")

    for row in df.itertuples():
        session.run(
            """
            MERGE (s:Season {season_name: $season})
            MERGE (g:Gameweek {season: $season, GW_number: $GW})
            MERGE (f:Fixture {season: $season, fixture_number: $fixture, kickoff_time : $kickoff})
            MERGE (t_home:Team {name: $home_team})
            MERGE (t_away:Team {name: $away_team})
            MERGE (p:Player {player_name: $name, player_element: $element})
            MERGE (pos:Position {name: $position})

            MERGE (s)-[:HAS_GW]->(g)
            MERGE (g)-[:HAS_FIXTURE]->(f)
            MERGE (f)-[:HAS_HOME_TEAM]->(t_home)
            MERGE (f)-[:HAS_AWAY_TEAM]->(t_away)
            MERGE (p)-[:PLAYS_AS]->(pos)
            MERGE (p)-[r:PLAYED_IN]->(f)
            SET r.minutes = $minutes,
                r.goals_scored = $goals_scored,
                r.assists = $assists,
                r.total_points = $total_points,
                r.bonus = $bonus,
                r.clean_sheets = $clean_sheets,
                r.goals_conceded = $goals_conceded,
                r.own_goals = $own_goals,
                r.penalties_saved = $penalties_saved,
                r.penalties_missed = $penalties_missed,
                r.yellow_cards = $yellow_cards,
                r.red_cards = $red_cards,
                r.saves = $saves,
                r.bps = $bps,
                r.influence = $influence,
                r.creativity = $creativity,
                r.threat = $threat,
                r.ict_index = $ict_index,
                r.form = $form
            """,
            season=row.season,
            GW=row.GW,
            fixture=row.fixture,
            home_team=row.home_team,
            away_team=row.away_team,
            name=row.name,
            element=row.element,
            position=row.position,
            kickoff=row.kickoff_time,
            minutes=row.minutes,
            goals_scored = row.goals_scored,
            assists=row.assists,
            total_points=row.total_points,
            bonus=row.bonus,
            clean_sheets=row.clean_sheets,
            goals_conceded=row.goals_conceded,
            own_goals=row.own_goals,
            penalties_saved=row.penalties_saved,
            penalties_missed=row.penalties_missed,
            yellow_cards=row.yellow_cards,
            red_cards=row.red_cards,
            saves=row.saves,
            bps=row.bps,
            influence=row.influence,
            creativity=row.creativity,
            threat=row.threat,
            ict_index=row.ict_index,
            form=row.form
        )
        print("Imported row:", row.Index)
    print("Data imported successfully.")



driver.close()