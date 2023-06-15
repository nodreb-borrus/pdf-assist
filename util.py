import math
import random
import hashlib
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter

spacy_splitter = SpacyTextSplitter(chunk_size=6969)
recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "\t", ". "],
    chunk_size=6969,
    chunk_overlap=0,
)


def round_up_to_next_even_thousand(number):
    rounded_number = math.ceil(number / 1000) * 1000
    # rounded_number = number + 500
    return rounded_number


def calculate_even_length(text_length, max_length):
    num_chunks = math.ceil(text_length / max_length)
    even_length = math.ceil(text_length / num_chunks)
    return even_length


# Take a list of docs, and return a new larger list where large docs above certain
# size are split into roughly even chunks
def map_splits(splitter_name, max_length, docs):
    splits = []
    for doc in docs:
        splits.extend(split_doc(splitter_name, max_length, doc))
    return splits


def split_doc(splitter_name, max_length, doc) -> List[str]:
    if splitter_name == "spacy":
        text_splitter = spacy_splitter
    else:
        text_splitter = recursive_splitter

    splits = []
    if len(doc) > max_length:
        # print(f"SEC {len(doc)} {round_up_to_next_even_thousand(len(doc))}")
        chunk_size = calculate_even_length(
            round_up_to_next_even_thousand(len(doc)), max_length
        )
        # print(f"EVEn {chunk_size}")
        text_splitter._chunk_size = chunk_size + 100
        docs = text_splitter.create_documents([doc])
        for d in docs:
            # print(len(d.page_content))
            splits.append((d.page_content))
    else:
        splits.append(doc)
    return splits


def map_lengths(docs):
    lengths = []
    for doc in docs:
        lengths.append(len(doc))
    return lengths


def map_tokens(llm, docs):
    lengths = []
    for doc in docs:
        lengths.append(llm.get_num_tokens(doc))
    return lengths


def calculate_checksum(text):
    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Convert the text to bytes and update the hash object
    md5_hash.update(text.encode("utf-8"))

    # Get the hexadecimal representation of the checksum
    checksum = md5_hash.hexdigest()

    return checksum


def fake_vector():
    vec = []
    for _ in range(1536):
        vec.append(random.uniform(0, 1))
    return vec


class FakeEmbeddings:
    def embed_documents(self, docs: List[str]):
        fakes = []
        for i in range(len(docs)):
            fakes.append(fake_vector())
        return fakes

    def embed_query(self, _: str):
        return fake_vector()


