import pandas as pd
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from community import community_louvain
from neo4j import GraphDatabase
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

# --- Configuration ---
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

THRESHOLD = 0.4
EMB_WEIGHT = 0.4
DIRECTOR_WEIGHT = 0.2
GENRE_WEIGHT = 0.2
STARS_WEIGHT = 0.2


# Load data
df = pd.read_csv("movies.csv")
df.fillna("", inplace=True)

# Normalize lists
df['Genre'] = df['Genre'].apply(lambda x: [g.strip() for g in x.split(',')])
df['Stars'] = df[['Star1', 'Star2', 'Star3', 'Star4']].values.tolist()

# Compute embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['Overview'].tolist(), show_progress_bar=True)

# Graph init
G = nx.Graph()

# Add nodes
for idx, row in df.iterrows():
    G.add_node(row['Series_Title'], **{
        'title': row['Series_Title'],
        'director': row['Director'],
        'genre': row['Genre'],
        'stars': row['Stars'],
        'rating': row['IMDB_Rating']
    })

# Compute hybrid similarity
for i in tqdm(range(len(df))):
    for j in range(i+1, len(df)):
        # Embedding similarity
        sim_emb = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        
        # Director match
        sim_director = 1.0 if df.iloc[i]['Director'] == df.iloc[j]['Director'] else 0.0
        
        # Genre Jaccard
        genres1 = set(df.iloc[i]['Genre'])
        genres2 = set(df.iloc[j]['Genre'])
        sim_genre = len(genres1 & genres2) / len(genres1 | genres2) if genres1 | genres2 else 0
        
        # Stars Jaccard
        stars1 = set(df.iloc[i]['Stars'])
        stars2 = set(df.iloc[j]['Stars'])
        sim_stars = len(stars1 & stars2) / len(stars1 | stars2) if stars1 | stars2 else 0

        # Weighted total
        sim_total = (
            EMB_WEIGHT * sim_emb +
            DIRECTOR_WEIGHT * sim_director +
            GENRE_WEIGHT * sim_genre +
            STARS_WEIGHT * sim_stars
        )

        if sim_total >= THRESHOLD:
            G.add_edge(df.iloc[i]['Series_Title'], df.iloc[j]['Series_Title'], weight=sim_total)

# Community detection
partition = community_louvain.best_partition(G, weight='weight')
nx.set_node_attributes(G, partition, 'community')

# Push to Neo4j
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")

    for node, data in tqdm(G.nodes(data=True), desc="Creating nodes"):
        session.run("""
            MERGE (m:Movie {title: $title})
            SET m.community = $community,
                m.director = $director,
                m.rating = $rating,
                m.genre = $genre,
                m.stars = $stars
        """, {
            'title': data['title'],
            'community': data['community'],
            'director': data['director'],
            'rating': data['rating'],
            'genre': ", ".join(data['genre']),
            'stars': ", ".join(data['stars']),
        })

    for u, v, data in tqdm(G.edges(data=True), desc="Creating edges"):
        session.run("""
            MATCH (m1:Movie {title: $u})
            MATCH (m2:Movie {title: $v})
            MERGE (m1)-[r:SIMILAR_TO]->(m2)
            SET r.weight = $weight
        """, {'u': u, 'v': v, 'weight': data['weight']})

driver.close()
