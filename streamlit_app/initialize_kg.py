"""
Knowledge Graph Initialization Script
Initializes the Neo4j knowledge graph with FPL data and creates vector embeddings.

This script performs the following steps:
1. Loads FPL data from CSV into Neo4j
2. Creates nodes (Season, Gameweek, Fixture, Team, Player, Position)
3. Creates relationships between nodes
4. Computes player embeddings using two models:
   - sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
   - sentence-transformers/all-mpnet-base-v2 (768 dimensions)
5. Creates vector indexes for semantic search

Run this script to initialize or re-initialize the knowledge graph.
"""

from neo4j import GraphDatabase
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import tqdm

# Load Neo4j configuration
config = {}
with open("config.txt", "r") as f:
    for line in f:
        key, value = line.strip().split("=", 1)
        config[key] = value

driver = GraphDatabase.driver(config["URI"], auth=(config["USERNAME"], config["PASSWORD"]))

print("=" * 80)
print("STEP 1: Importing FPL data into Neo4j knowledge graph")
print("=" * 80)

# Load the preprocessed FPL data
df = pd.read_csv("fpl_graph_final.csv", header=0)

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
            'team_x': row.team_x,
            'opp_team_name': row.opp_team_name,
            'name': row.name,
            'element': row.element,
            'code': row.code,
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

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]

        session.run(
            """
            UNWIND $batch as row

            // === NODES ===
            MERGE (s:Season {season_name: row.season})
            MERGE (g:Gameweek {season: row.season, GW_number: row.GW})
            MERGE (f:Fixture {season: row.season, fixture_number: row.fixture})
                SET f.kickoff_time = row.kickoff

            // Teams
            MERGE (t_home:Team {name: row.home_team})
            MERGE (t_away:Team {name: row.away_team})
            MERGE (t_player:Team {name: row.team_x})
            MERGE (t_opp:Team {name: row.opp_team_name})

            // Player
            MERGE (p:Player {
                player_name: row.name,
                code: row.code
            })


            // Position
            MERGE (pos:Position {name: row.position})


            // === RELATIONSHIPS ===

            // Season → GW
            MERGE (s)-[:HAS_GW]->(g)

            // GW → Fixture
            MERGE (g)-[:HAS_FIXTURE]->(f)

            // Fixture teams
            MERGE (f)-[:HAS_HOME_TEAM]->(t_home)
            MERGE (f)-[:HAS_AWAY_TEAM]->(t_away)

            // Player → Position
            MERGE (p)-[:PLAYS_AS]->(pos)

            // Player → Team (career/season relationship)
            MERGE (p)-[:PLAYS_FOR {season: row.season}]->(t_player)

            // Opponent link (useful for queries)
            MERGE (p)-[:PLAYED_AGAINST]->(t_opp)

            // Player → Fixture performance stats
            MERGE (p)-[r:PLAYED_IN]->(f)
            SET
                r.minutes = row.minutes,
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

print("\n" + "=" * 80)
print("STEP 2: Fetching aggregated player statistics for embedding generation")
print("=" * 80)

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

    df_players = pd.DataFrame(result.data())

print(f"Fetched {len(df_players)} players from Neo4j")

print("\n" + "=" * 80)
print("STEP 3: Building textual descriptions for each player")
print("=" * 80)

# Build a textual description per player for the text model
def build_player_description(row):
    return (
        f"Player: {row['name']}, "
        f"Position: {row['position']}, "
        f"Team: {row['team']}, "
        f"Goals: {row['sum_goals']}, "
        f"Assists: {row['sum_assists']}, "
        f"Total points: {row['sum_points']}, "
        f"Minutes played: {row['sum_minutes']}, "
        f"Clean sheets: {row['sum_clean_sheets']}, "
        f"Influence: {row['avg_influence']:.1f}, "
        f"Creativity: {row['avg_creativity']:.1f}, "
        f"Threat: {row['avg_threat']:.1f}, "
        f"ICT index: {row['avg_ict_index']:.1f}."
    )

df_players["description"] = df_players.apply(build_player_description, axis=1)

print("Sample description:")
print(df_players["description"].iloc[0])

print("\n" + "=" * 80)
print("STEP 4: Computing embeddings with sentence-transformers models")
print("=" * 80)

# Load embedding models
print("Loading all-MiniLM-L6-v2 model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Loading all-mpnet-base-v2 model...")
model_v2 = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Compute embeddings for all players
print("\nGenerating embeddings with all-MiniLM-L6-v2 (384 dimensions)...")
text_embeddings = model.encode(
    df_players["description"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nGenerating embeddings with all-mpnet-base-v2 (768 dimensions)...")
text_embeddings_v2 = model_v2.encode(
    df_players["description"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Attach to dataframe as list-of-floats
df_players["embedding_text_l6_v2"] = [vec.astype(float).tolist() for vec in text_embeddings]
df_players["embedding_text_mpnet_v2"] = [vec.astype(float).tolist() for vec in text_embeddings_v2]

print(f"\nEmbedding dimensions:")
print(f"  - all-MiniLM-L6-v2: {len(df_players['embedding_text_l6_v2'].iloc[0])}")
print(f"  - all-mpnet-base-v2: {len(df_players['embedding_text_mpnet_v2'].iloc[0])}")

print("\n" + "=" * 80)
print("STEP 5: Writing embeddings to Neo4j Player nodes")
print("=" * 80)

def write_embeddings(tx, code, emb_text):
    tx.run(
        """
        MATCH (p:Player {code: $code})
        SET p.embedding_text_l6_v2 = $embedding_text_l6_v2
        """,
        code=code,
        embedding_text_l6_v2=emb_text
    )

def write_embeddings_v2(tx, code, emb_text_v2):
    tx.run(
        """
        MATCH (p:Player {code: $code})
        SET p.embedding_text_mpnet_v2 = $embedding_text_mpnet_v2
        """,
        code=code,
        embedding_text_mpnet_v2=emb_text_v2
    )
def write_desc(tx, code, desc):
    tx.run(
        """
        MATCH (p:Player {code: $code})
        SET p.description= $desc
        """,
        code=code,
        desc=desc
    )

with driver.session() as session:
    # Iterate through players
    for _, row in tqdm.tqdm(df_players.iterrows(), total=len(df_players), desc="Writing embeddings"):
        session.execute_write(
            write_embeddings,
            row["code"],
            row["embedding_text_l6_v2"]
        )
        session.execute_write(
            write_embeddings_v2,
            row["code"],
            row["embedding_text_mpnet_v2"]
        )
        session.execute_write(
            write_desc,
            row["code"],
            row["description"]
        )


print("Embeddings successfully written to Neo4j!")

print("\n" + "=" * 80)
print("STEP 6: Creating vector indexes for similarity search")
print("=" * 80)

def create_indexes(tx):
    # Text embedding index with all-MiniLM-L6-v2 (dimension = 384)
    tx.run("""
        CREATE VECTOR INDEX playerEmbeddingTextIndexL6V2
        IF NOT EXISTS
        FOR (p:Player) ON (p.embedding_text_l6_v2)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: "cosine"
            }
        };
    """)

    # Text embedding index with all-mpnet-base-v2 (dimension = 768)
    tx.run("""
        CREATE VECTOR INDEX playerEmbeddingTextIndexMpnetV2
        IF NOT EXISTS
        FOR (p:Player) ON (p.embedding_text_mpnet_v2)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: "cosine"
            }
        };
    """)

with driver.session() as session:
    session.execute_write(create_indexes)

print("Vector indexes created successfully!")
print("  - playerEmbeddingTextIndexL6V2 (384 dimensions)")
print("  - playerEmbeddingTextIndexMpnetV2 (768 dimensions)")

driver.close()

print("\n" + "=" * 80)
print("INITIALIZATION COMPLETE!")
print("=" * 80)
print("\nThe knowledge graph has been initialized with:")
print("  - Nodes: Season, Gameweek, Fixture, Team, Player, Position")
print("  - Relationships: HAS_GW, HAS_FIXTURE, HAS_HOME_TEAM, HAS_AWAY_TEAM,")
print("                  PLAYS_AS, PLAYS_FOR, PLAYED_AGAINST, PLAYED_IN")
print(f"  - {len(df_players)} player embeddings (2 models)")
print("  - 2 vector indexes for semantic similarity search")
print("\nYou can now run the Streamlit app!")