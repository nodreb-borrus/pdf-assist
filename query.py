from sqlmodel import Session, select
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from database import Book, Chapter, Section, Query, engine

embeddings = OpenAIEmbeddings()

# Query the sections for similarity
# book_id ignored atm, only one book
def query_embeddings(book_id, query_text):
    with Session(engine) as session:
        # Look for identical query, to use the existing embedding
        query_finder = select(Query).where(Query.text == query_text)
        results = session.exec(query_finder)
        query = results.first()

        if not query:
            vector = embeddings.embed_query(query_text)
            print(f"Fetched query embedding")
            query = Query(book_id=book_id, text=query_text, embedding=vector)
            session.add(query)
            session.commit()

        select_q = (
            select(
                Section,
                Section.embedding.cosine_distance(query.embedding)
            )
            .order_by(Section.embedding.cosine_distance(query.embedding))
            .limit(10)
        )
        results = session.exec(select_q)
        for result in results:
            print(f"Match score: {result[1]}")
            print(result[0].text)
            print()


# query_embeddings(1, "Is a man mechanical or conscious?")
query_embeddings(1, "What is the political situation in Russia?")
