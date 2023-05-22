import os
import re
import math
import fitz
import random
from typing import List
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sqlmodel import Session
from dotenv import load_dotenv

load_dotenv()

from database import Book, Chapter, Section, engine


def flags_decomposer(flags):
    """Make font flags human readable."""
    l = []
    if flags & 2**0:
        l.append("superscript")
    if flags & 2**1:
        l.append("italic")
    if flags & 2**2:
        l.append("serifed")
    else:
        l.append("sans")
    if flags & 2**3:
        l.append("monospaced")
    else:
        l.append("proportional")
    if flags & 2**4:
        l.append("bold")
    return ", ".join(l)


def get_block_text(block):
    text = ""
    for l in block["lines"]:  # iterate through the text spans
        text = text + get_line_text(l)
    return text


def get_line_text(line):
    text = ""
    for s in line["spans"]:  # iterate through the text spans
        text = text + s["text"]
    return text


def has_footnote(page):
    blocks = page.get_text("dict", flags=11)["blocks"]
    for b in blocks:  # iterate through the text blocks
        for l in b["lines"]:  # iterate through the text lines
            # Regular text is about 8 and superscripts 5+
            if l["spans"][0]["size"] < 6:
                # print(page.number, get_line_text(l))
                # print(page.number, get_block_text(b))
                return True
            else:
                return False


# Tries to find section boundaries, where there is an extra blank line in text
def detect_sections(page):
    words = page.get_text("words")  # Get individual words with positions

    # Analyze word positions and spacing
    paragraphs = []
    current_paragraph = []
    prev_word_y = None

    for word in words:
        word_text = word[4]
        word_y = word[3]

        if (
            prev_word_y is not None and abs(word_y - prev_word_y) > 16
        ):  # Adjust the threshold as needed
            # Detected significant vertical gap, start new paragraph
            if current_paragraph:
                paragraphs.append(" ".join(current_paragraph))
            current_paragraph = []

        current_paragraph.append(word_text)
        prev_word_y = word_y

    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph))

    return paragraphs


def chapter_heading(text):
    if re.search(r"chapter", text, re.IGNORECASE) and len(text) < 20:
        return True
    else:
        return False


def incomplete_text(text):
    if (
        text[-3:] == ". \n"
        or text[-3:] == '" \n'
        or text[-3:] == "' \n"
        or text[-3:] == "? \n"
        or text[-3:] == "! \n"
        or text[-3:] == ": \n"
        or text[-2:] == ".\n"
        or chapter_heading(text)
    ):
        return False
    else:
        return True


def merge_block_texts(first, second):
    first_non_break = first[:-1]
    if first_non_break[-1] == "-":
        first_non_break = first_non_break[:-1]
    return first_non_break + second


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
def map_splits(max_length, docs):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "\t", ". "], chunk_size=max_length, chunk_overlap=0
    )

    splits = []
    for doc in docs:
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


# Specify the path to your PDF file
pdf_path = "./data/OUSPENSKY, P.D. - In Search of the Miraculous.pdf"
book_name = os.path.splitext(os.path.basename(pdf_path))[0]

llm = OpenAI(model_name="gpt-3.5-turbo")  # pyright:ignore

with fitz.open(pdf_path) as doc:
    # Get chapter page numbers from toc
    chapter_page_starts = []
    toc = doc.get_toc()
    if toc:
        for entry in toc:
            level = entry[0]
            title = entry[1]
            page_num = entry[2]
            print(f"{level} {title}: Page {page_num}")
            chapter_page_starts.append(page_num)

    # Calculate lengths of each chapter using toc
    chapter_lengths = []
    for i, page_num in enumerate(chapter_page_starts):
        if i == len(chapter_page_starts) - 1:
            chapter_lengths.append(len(doc) - chapter_page_starts[i])
        else:
            chapter_lengths.append(chapter_page_starts[i + 1] - page_num)

    print(chapter_page_starts)
    print(chapter_lengths)

    # Build up chapers using entire page text
    full_chapters: List[str] = []
    working_chapter = []

    # Build up chapters using text blocks
    block_list = []
    block_chapters: List[List[str]] = []
    working_block_chapter = []

    # Build up sections
    section_list = []
    section_chapters = []
    working_section_chapter = []

    # Main loop, go through page by page
    for i, page in enumerate(doc):
        # Pop off current chapter start page once reached
        page_num = i + 1
        if chapter_page_starts and page_num == chapter_page_starts[0]:
            start = chapter_page_starts.pop(0)

        # Basic page text to build chapters
        page_text = page.get_text()
        working_chapter.append(page_text)

        # Build up using blocks to remove footnotes, and merge paragraphs spanning pages
        page_blocks = page.get_text("blocks")
        for j, block in enumerate(page_blocks):
            # Omit image blocks for now
            if block[6] == 1:
                continue
            # If we have blocks already, and last one cut off by end of page
            # Then merge first block of this page with it
            elif (
                block_list
                and working_block_chapter
                and j == 0
                and incomplete_text(block_list[-1])
            ):
                # print(working_block_chapter[-1][-40:].encode("utf-8"))
                new_previous_block = merge_block_texts(block_list[-1], block[4])
                block_list[-1] = new_previous_block
                working_block_chapter[-1] = new_previous_block
            # For typical blocks just add them to the lists
            else:
                block_list.append(block[4])
                working_block_chapter.append(block[4])

        sections = detect_sections(page)
        # if len(sections) > 4:
        #     print("Page", page_num)
        #     for sec in sections:
        #         print(len(sec))
        for j, section in enumerate(sections):
            # Continue sections across pages as with blocks
            if (
                section_list
                and working_section_chapter
                and j == 0
                and incomplete_text(section_list[-1] + " \n")
                # Combine multiple small sections on one page
            ) or (
                len(sections) > 4
                and j != 0
                and len(section_list[-1]) < 650
                and len(section) < 350
            ):
                new_previous_section = merge_block_texts(
                    section_list[-1] + " \n", section
                )
                section_list[-1] = new_previous_section
                working_section_chapter[-1] = new_previous_section
            # Omit leftover short sections
            elif len(section) > 75:
                section_list.append(section)
                working_section_chapter.append(section)

        # When on last page of the chapter/book, wrap up working chapters
        if (
            chapter_page_starts and page_num + 1 == chapter_page_starts[0]
        ) or page_num == len(doc):
            final_chapter = "\n".join(working_chapter)
            full_chapters.append(final_chapter)
            working_chapter = []

            block_chapters.append(working_block_chapter)
            working_block_chapter = []

            section_chapters.append(working_section_chapter)
            working_section_chapter = []

    # Use chapters after generating
    # for i, chapter in enumerate(full_chapters):
    #     lent = len(chapter)
    #     toks = llm.get_num_tokens(chapter)
    #     page_len = chapter_lengths[i]
    #     print(f"{len(chapter)} {chapter_lengths[i]} {len(chapter)/chapter_lengths[i]}")
    #     print(f"{toks} {toks/page_len}")
    #     print(chapter[:65].encode("utf-8"))

    # Count blocks in each chapter, show avg blocks per page
    # for i, chapter in enumerate(block_chapters):
    #     print(
    #         f"Block Chapter {i}: {len(chapter)} blocks, {len(chapter)/chapter_lengths[i]} block per page"
    #     )

    # for b in block_chapters[1]:
    #     print(b)

    # section_lengths = []
    # for i, s in enumerate(section_list):
    #     tokens = llm.get_num_tokens(s)
    #     section_lengths.append(len(s))
    # print(sorted(section_lengths))

    print(f"Chapters: {len(full_chapters)}")

    print(f"Blocks: {len(block_list)}")

    print(f"Sections: {len(section_list)}")


# Load the book
# loader = PyMuPDFLoader(pdf_path)
# pages = loader.load(flags=fitz.TEXT_PRESERVE_WHITESPACE)

# print(f"Pages: {len(pages)}")

# page_index = 138
# page1 = pages[page_index].page_content
# print(page1)

# print(page1.encode("utf-8"))
# Combine the pages, and replace the tabs with spaces
# text = ""

# for page in pages:
#     text += page.page_content

# text = text.replace("\t", " ")


# Detect paragraphs on the specific page
# paragraphs = detect_sections(pdf_path, page_index)

# Print the detected paragraphs
# for paragraph in paragraphs:
#     print(paragraph)
#     print()

# num_tokens = llm.get_num_tokens(page1)

# print(f"This book has {num_tokens} tokens in it")


# split_secs = map_splits(5000, section_list)
# section_lengths = map_lengths(split_secs)
# section_tokens = map_tokens(llm, split_secs)
# print(section_lengths)
# print(section_tokens)
# print(sorted(section_lengths))
# print(len(section_list))
# print(len(section_lengths))

for chap in section_chapters:
    print(len(chap))
    print(map_lengths(chap))
    split_chap = map_splits(5000, chap)
    print(map_lengths(split_chap))


embeddings = OpenAIEmbeddings()
# print(f"Contents Sections")
# for i, sec in enumerate(split_secs[:19]):
#     print(f"{i} {sec}")

# vectors = embeddings.embed_documents([split_secs[322]])
# print(vectors)


def fake_vector():
    vec = []
    for _ in range(1536):
        vec.append(random.uniform(0, 1))
    return vec

def store_book_with_embeddings():
    with Session(engine) as session:
        # Create the book
        book = Book(name=book_name)
        session.add(book)
        session.commit()

        # Create all the chapters
        ordered_chapters = []
        for i, chapter in enumerate(section_chapters):
            if i == 0:  # Skip title page
                continue
            elif i == 1:
                chap = Chapter(name="CONTENTS", book_id=book.id)
            else:
                # 3rd item is Chapter 1
                chap_name = str(i - 1)
                chap = Chapter(name=chap_name, book_id=book.id)

            session.add(chap)
            ordered_chapters.append(chap)

        session.commit()

        # Create all the text sections
        for i, chapter in enumerate(ordered_chapters):
            # Skip empty title page
            sections = section_chapters[i + 1]
            vectors = embeddings.embed_documents(sections)
            print(f"Chapter {chapter.name} embeddings fetched.")
            for j, section in enumerate(sections):
                sec = Section(
                    chapter_id=chapter.id,
                    chapter_index=j,
                    text=section,
                    embedding=vectors[j],
                )
                session.add(sec)

        session.commit()


# Actually fetch embeddings into db
# Do only once
# store_book_with_embeddings()
