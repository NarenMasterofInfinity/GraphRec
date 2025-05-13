import streamlit as st
from neo4j import GraphDatabase
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

# --- Configuration ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

st.set_page_config(page_title="Movie Recommender", layout="wide")
# Neo4j driver
@st.cache_resource
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

driver = get_driver()

# Load all movie titles for autocomplete
@st.cache_data(ttl=3600)
def load_titles():
    with driver.session() as session:
        result = session.run("MATCH (m:Movie) RETURN m.title AS title ORDER BY title")
        return [record["title"] for record in result]

titles = load_titles()

# Fuzzy-match user input to suggestions
def suggest_title(input_title):
    if not input_title: return []
    matches = process.extract(input_title, titles, limit=10)
    return [m[0] for m in matches if m[1] > 60]

# Query functions
@st.cache_data(ttl=600)
def query_genre_recs(title):
    cypher = '''
    MATCH (m:Movie {title:$title})-[:HAS_GENRE]->(g:Genre)<-[:HAS_GENRE]-(rec:Movie)
    WHERE rec.title <> $title
    WITH rec, collect(g.name) AS shared_genres
    RETURN rec.title AS title, shared_genres, rec.imdb_rating AS rating, rec.overview AS overview
    ORDER BY size(shared_genres) DESC, rating DESC
    LIMIT 5
    '''
    with driver.session() as session:
        return session.run(cypher, title=title).data()

@st.cache_data(ttl=600)
def query_people_recs(title):
    cypher = '''
    MATCH (m:Movie {title:$title})-[:DIRECTED_BY|:STARRING]->(p:Person)<-[:DIRECTED_BY|:STARRING]-(rec:Movie)
    WHERE rec.title <> $title
    WITH rec, collect(DISTINCT p.name) AS shared_people
    RETURN rec.title AS title, shared_people, rec.imdb_rating AS rating, rec.overview AS overview
    ORDER BY size(shared_people) DESC, rating DESC
    LIMIT 5
    '''
    with driver.session() as session:
        return session.run(cypher, title=title).data()

@st.cache_data(ttl=600)
def query_community_recs(title):
    cypher = '''
    MATCH (m:Movie {title:$title})
    WITH m.community AS comm
    MATCH (rec:Movie {community:comm})
    WHERE rec.title <> $title
    RETURN rec.title AS title, rec.imdb_rating AS rating, rec.overview AS overview
    LIMIT 5
    '''
    with driver.session() as session:
        return session.run(cypher, title=title).data()

@st.cache_data(ttl=600)
def query_plot_recs(title):
    cypher = '''
    MATCH (m:Movie {title:$title})-[r:SIMILAR_TO|SIMILAR_OVERVIEW]->(rec:Movie)
    WHERE rec.title <> $title
    RETURN rec.title AS title, r.weight AS score, rec.imdb_rating AS rating, rec.overview AS overview
    ORDER BY score DESC
    LIMIT 5
    '''
    with driver.session() as session:
        return session.run(cypher, title=title).data()

# Streamlit UI

st.title("Graph-Based Movie Recommender")

user_input = st.text_input("Enter movie title:")

suggestions = suggest_title(user_input)
if suggestions:
    choice = st.selectbox("Did you mean:", suggestions)
    if st.button("Recommend"):
        # genre-based
        st.header("Top 5 by Shared Genres")
        for rec in query_genre_recs(choice):
            with st.expander(f"{rec['title']} — Rating: {rec['rating']}"):
                st.write("**Shared Genres:** ", ", ".join(rec['shared_genres']))
                st.write(rec['overview'])

        # people-based
        st.header("Top 5 by Shared Director/Stars")
        for rec in query_people_recs(choice):
            with st.expander(f"{rec['title']} — Rating: {rec['rating']}"):
                st.write("**Shared People:** ", ", ".join(rec['shared_people']))
                st.write(rec['overview'])

        # community-based
        st.header("Community Recommendations")
        for rec in query_community_recs(choice):
            with st.expander(f"{rec['title']} — Rating: {rec['rating']}"):
                st.write(rec['overview'])

        # plot-based
        st.header("Top 5 by Plot Similarity")
        for rec in query_plot_recs(choice):
            with st.expander(f"{rec['title']} — Score: {rec['score']:.2f}"):
                st.write(rec['overview'])
else:
    st.info("Start typing to see suggestions...")
