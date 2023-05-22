import os
from typing import List, Optional
from sqlmodel import Field, SQLModel, Column, create_engine
from pgvector.sqlalchemy import Vector
from langchain.vectorstores.pgvector import PGVector

class Book(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str


class Chapter(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    book_id: Optional[int] = Field(default=None, foreign_key="book.id")
    name: str


class Section(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chapter_id: Optional[int] = Field(default=None, foreign_key="chapter.id")
    chapter_index: int
    text: str
    embedding: List[float] = Field(sa_column=Column(Vector(1536)))


class Query(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    book_id: Optional[int] = Field(default=None, foreign_key="book.id")
    text: str
    embedding: List[float] = Field(sa_column=Column(Vector(1536)))


CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "localhost"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
    user=os.environ.get("PGVECTOR_USER", "postgres"),
    password=os.environ.get("PGVECTOR_PASSWORD", "postgres"),
)


# Manually:
# CREATE DATABASE pdf_assist;
# CREATE EXTENSION vector;
engine = create_engine(CONNECTION_STRING)
# Don't use after fetching embeddings (NOW)
# SQLModel.metadata.drop_all(engine)
SQLModel.metadata.create_all(engine)


