import os
from typing import List, Optional, TypeVar, Generic
from datetime import datetime
import sqlalchemy
from sqlmodel import (
    Field,
    SQLModel,
    Column,
    Relationship,
    DateTime,
    create_engine,
    func,
    select,
)
from pgvector.sqlalchemy import Vector
from langchain.vectorstores.pgvector import PGVector

T = TypeVar("T")


class CRUD(Generic[T]):
    # Hack to make mixins work with SQLModel
    # https://github.com/tiangolo/sqlmodel/pull/256#issuecomment-1112188647
    __config__ = None

    @classmethod
    def get(cls, session, **kwargs) -> Optional[T]:
        q = select(cls)
        for key, value in kwargs.items():
            q = q.where(getattr(cls, key) == value)
        return session.exec(q).first()

    @classmethod
    def get_all(cls, session, **kwargs) -> List[T]:
        q = select(cls).order_by(cls.id)
        for key, value in kwargs.items():
            q = q.where(getattr(cls, key) == value)
        return session.exec(q).all()

    @classmethod
    def create(cls, session, **kwargs) -> T:
        obj = cls(**kwargs)
        session.add(obj)
        session.commit()
        session.refresh(obj)
        return obj

    @classmethod
    def get_or_create(cls, session, **kwargs) -> T:
        obj = cls.get(session, **kwargs)
        if not obj:
            obj = cls.create(session, **kwargs)
        return obj


class Book(SQLModel, CRUD["Book"], table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str


class Chapter(SQLModel, CRUD["Chapter"], table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    book_id: Optional[int] = Field(default=None, foreign_key="book.id")
    name: str


# Everytime we change the logic for making sections, save a new batch
class SectionBatch(SQLModel, CRUD["SectionBatch"], table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tag: Optional[str]
    created_at: Optional[datetime] = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )


# Represents a logical section, smaller than a chapter.
# For example when text is broken by an extra newline and changes context


# v1
# class Section(SQLModel, table=True):
#     id: Optional[int] = Field(default=None, primary_key=True)
#     chapter_id: Optional[int] = Field(default=None, foreign_key="chapter.id")
#     chapter_index: int
#     text: str
#     embedding: List[float] = Field(sa_column=Column(Vector(1536)))


class SectionChunk(SQLModel, CRUD["SectionChunk"], table=True):
    section_id: Optional[int] = Field(
        default=None, foreign_key="sectionl.id", primary_key=True
    )
    chunk_id: Optional[int] = Field(
        default=None, foreign_key="chunk.id", primary_key=True
    )
    index: int


# v2
class SectionL(SQLModel, CRUD["SectionL"], table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chapter_id: Optional[int] = Field(
        default=None, foreign_key="chapter.id", nullable=False
    )
    batch_id: Optional[int] = Field(
        default=None, foreign_key="sectionbatch.id", nullable=False
    )
    chunks: List["Chunk"] = Relationship(link_model=SectionChunk)
    index: int
    text: str
    checksum: str


class ChunkBatch(SQLModel, CRUD["ChunkBatch"], table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tag: Optional[str]
    created_at: Optional[datetime] = Field(
        sa_column=Column(
            DateTime(timezone=True), server_default=func.now(), nullable=False
        )
    )


class Chunk(SQLModel, CRUD["Chunk"], table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    batch_id: Optional[int] = Field(
        default=None, foreign_key="chunkbatch.id", nullable=False
    )
    text: str
    checksum: str
    embedding: List[float] = Field(sa_column=Column(Vector(1536)))


class Query(SQLModel, CRUD["Query"], table=True):
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
engine = None
try:
    engine = create_engine(CONNECTION_STRING)
    # Don't use after fetching embeddings (NOW)
    # SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)
except sqlalchemy.exc.OperationalError as e:
    print("Could not connect to database.")
