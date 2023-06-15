from sqlmodel import Session, select
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

from database import ChunkBatch, SectionChunk, SectionL, Chunk, Query, engine

llm = OpenAI(model_name="gpt-3.5-turbo")  # pyright:ignore
embeddings = OpenAIEmbeddings()  # pyright:ignore

# Query the sections for similarity
# book_id ignored atm, only one book
def query_embeddings(book_id, query_text):
    with Session(engine) as session:
        # Look for identical query, to use the existing embedding
        query = Query.get(session, text=query_text)

        if not query:
            vector = embeddings.embed_query(query_text)
            print(f"Fetched query embedding")
            query = Query.create(session, book_id=book_id, text=query_text, embedding=vector)

        select_q = (
            select(
                SectionL,
                SectionChunk,
                ChunkBatch,
                Chunk,
                Chunk.embedding.cosine_distance(query.embedding)
            ) # pyright: ignore
            .where(ChunkBatch.tag == "SmallSecFixes-Spacy-2000-05-26")
            .where(SectionL.chapter_id != 1) # Hack to exclude table of contents
            .where(Chunk.batch_id == ChunkBatch.id)
            .where(SectionChunk.chunk_id == Chunk.id)
            .where(SectionL.id == SectionChunk.section_id)
            .order_by(Chunk.embedding.cosine_distance(query.embedding))
            .limit(20)
        )
        results = session.exec(select_q)
        for result in results:
            print(f"Match score: {result[4]}")
            print(f"Tokens: {llm.get_num_tokens(result[3].text)}")
            print(result[3].text.encode("utf-8"))
            print()


query_embeddings(1, "Is a man mechanical or conscious?")
# query_embeddings(1, "What is the political situation in Russia?")
# query_embeddings(1, "What is the role of the moon in the cosmos and what effect does it have on life?")
