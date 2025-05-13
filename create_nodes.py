

import pandas as pd
import itertools
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util

NEO4J_URI      = "neo4j+s://926182d4.databases.neo4j.io"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "D7JTWEchFDbZS6oI0_TyKYS1sEpWMJQ1GozX8r39ysA"

MOVIES_CSV     = "movies.csv"
SIM_THRESHOLD  = 0.4


W_OVERVIEW     = 0.4
W_DIRECTOR     = 0.2
W_GENRE        = 0.2
W_STARS        = 0.2

df = pd.read_csv(MOVIES_CSV).fillna("")

df["Genres"] = df["Genre"].apply(lambda s: [g.strip() for g in s.split(",") if g.strip()])
df["Stars"]  = df[["Star1","Star2","Star3","Star4"]].apply(lambda row: [x for x in row if x], axis=1)

titles = df["Series_Title"].tolist()


model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["Overview"].tolist(), convert_to_tensor=True)


edges = []
for i, j in itertools.combinations(range(len(df)), 2):

    ov_sim = util.cos_sim(embeddings[i], embeddings[j]).item()
    if ov_sim < SIM_THRESHOLD:
        continue

    # director bonus
    dir_sim = 1.0 if df.at[i,"Director"] == df.at[j,"Director"] and df.at[i,"Director"] else 0.0
    # genre jaccard
    set_gi, set_gj = set(df.at[i,"Genres"]), set(df.at[j,"Genres"])
    genre_sim = len(set_gi & set_gj) / len(set_gi | set_gj) if (set_gi | set_gj) else 0.0
    # stars jaccard
    set_si, set_sj = set(df.at[i,"Stars"]), set(df.at[j,"Stars"])
    stars_sim = len(set_si & set_sj) / len(set_si | set_sj) if (set_si | set_sj) else 0.0

   
    score = (
        W_OVERVIEW * ov_sim +
        W_DIRECTOR * dir_sim +
        W_GENRE * genre_sim +
        W_STARS * stars_sim
    )

    edges.append({"title1": title_i, "title2": title_j, "weight": float(score)})


driver = GraphDatabase.driver(NEO4J_URI,
                              auth=(NEO4J_USER, NEO4J_PASSWORD))

def create_constraints(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Movie) REQUIRE m.title IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name  IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Genre)  REQUIRE g.name  IS UNIQUE")

def ingest_movies(tx, records):
    tx.run("""
    UNWIND $rows AS row
    MERGE (m:Movie {title: row.Series_Title})
      SET m.released_year = toInteger(row.Released_Year),
          m.certificate   = row.Certificate,
          m.runtime       = row.Runtime,
          m.imdb_rating   = toFloat(row.IMDB_Rating),
          m.meta_score    = toInteger(row.Meta_score),
          m.no_of_votes   = toInteger(row.No_of_Votes),
          m.gross         = toFloat(row.Gross),
          m.overview      = row.Overview
    FOREACH (gName IN row.Genres |
      MERGE (g:Genre {name: gName})
      MERGE (m)-[:HAS_GENRE]->(g)
    )
    MERGE (d:Person {name: row.Director})
    MERGE (m)-[:DIRECTED_BY]->(d)
    FOREACH (star IN row.Stars |
      MERGE (s:Person {name: star})
      MERGE (m)-[:STARRING]->(s)
    );
    """, rows=records)

def ingest_edges(tx, edges):
    tx.run("""
    UNWIND $edges AS e
    MATCH (a:Movie {title: e.title1}), (b:Movie {title: e.title2})
    MERGE (a)-[r:SIMILAR_OVERVIEW]-(b)
      SET r.weight = e.weight
    """, edges=edges)

with driver.session() as session:

    session.write_transaction(create_constraints)
  
    session.write_transaction(ingest_movies, df.to_dict("records"))

    session.write_transaction(ingest_edges, edges)

driver.close()
print(f"Ingested {len(df)} movies and {len(edges)} similarity edges into Neo4j.")
