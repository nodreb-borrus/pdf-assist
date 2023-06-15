import os
import re
import fitz
from typing import List
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from pydantic import BaseModel
from sqlmodel import Session
from dotenv import load_dotenv

load_dotenv()

from util import FakeEmbeddings, map_lengths, map_tokens, split_doc, map_splits, calculate_checksum
from database import (
    Book,
    Chapter,
    Chunk,
    ChunkBatch,
    SectionBatch,
    SectionChunk,
    SectionL,
    engine,
)


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


# Specify the path to your PDF file
pdf_path = "./data/OUSPENSKY, P.D. - In Search of the Miraculous.pdf"
# pdf_path = "./data/GURDJIEFF, G.I. - Beelzebub's Tales to His Grandson.pdf"
book_name = os.path.splitext(os.path.basename(pdf_path))[0]

llm = OpenAI(model_name="gpt-3.5-turbo")  # pyright:ignore


class TocChapter(BaseModel):
    level: int
    title: str
    page: int
    length: int


def table_of_contents(fitz_doc):
    return []


with fitz.open(pdf_path) as doc:
    # Get chapter page numbers from toc
    chapter_page_starts = []
    toc = doc.get_toc()
    if toc:
        for entry in toc:
            level = entry[0]
            title = entry[1]
            page_num = entry[2]
            # print(f"{level} {title}: Page {page_num}")
            chapter_page_starts.append(page_num)

    # Calculate lengths of each chapter using toc
    chapter_lengths = []
    for i, page_num in enumerate(chapter_page_starts):
        if i == len(chapter_page_starts) - 1:
            chapter_lengths.append(len(doc) - chapter_page_starts[i])
        else:
            chapter_lengths.append(chapter_page_starts[i + 1] - page_num)

    # print(chapter_page_starts)
    # print(chapter_lengths)

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

    # Signal to the next page that previous page ended with short section so merge
    force_merge = False

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

        # BIG NOTES ENERGY
        # DONE We are fixing broken sentences across pages!
        # DONE 3+ small sections on a page: merge (how to merge also with adjacent?, these are usually figures)
        # MAYBE  ^ try also just merging them all up to previous regardless of previous length
        # DONE Small section at beginning of page: prob merge with previous page section
        # DONE Small section at end of page: prob merge with following page
        # 2HARD Images: merge before and after sections (Chapter 5)
        # Small section ending with : ; etc.: merge down
        # Pages ending on sentence, sections not small, but should be merged still
        # Legit separate sections which are just small and lose surrounding context
        # Page 240: small sections with images should be merged to both prev and next pages
        # TODO Need to deal with footnotes: breaks broken line detection

        small_size = 650
        smaller_size = 450  # Don't go smaller, it will consume chapter labels
        for j, section in enumerate(sections):
            # Continue sections across pages as with blocks
            if (
                (
                    section_list
                    and working_section_chapter
                    and j == 0
                    and (
                        # Text broken across pages
                        incomplete_text(section_list[-1] + " \n")
                        # Top section of page is short
                        or len(section) < small_size
                    )
                )
                # Combine multiple small sections on one page
                or (
                    len(sections) > 2
                    and j != 0
                    and len(section_list[-1]) < small_size
                    and len(section) < smaller_size
                )
                or force_merge
            ):
                new_previous_section = merge_block_texts(
                    section_list[-1] + " \n", section
                )
                section_list[-1] = new_previous_section
                working_section_chapter[-1] = new_previous_section
                force_merge = False
            else:
                # if len(sections) - 1 == j and len(section) < 650:
                #     force_merge = True
                # print(page_num)
                section_list.append(section)
                working_section_chapter.append(section)

            # If last section on page is short, merge with next page
            # or if a section ends with colon, merge with next section
            if (len(sections) - 1 == j and len(section) < 650) or (
                section.strip()[-1] == ":" or section.strip()[-1] == ";"
            ):
                force_merge = True

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

            force_merge = False

    # Use chapters after generating
    # Print start text and length/token info
    # for i, chapter in enumerate(full_chapters):
    #     lent = len(chapter)
    #     toks = llm.get_num_tokens(chapter)
    #     page_len = chapter_lengths[i]

    #     print(chapter[:65].encode("utf-8"))
    #     print(f"Length: {len(chapter)} Page length: {chapter_lengths[i]} Length/page: {len(chapter)/chapter_lengths[i]}")
    #     print(f"Tokens: {toks} Token/page: {toks/page_len}")
    #     print()

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
    # print(section_lengths)
    # print(sorted(section_lengths))

    # TODO remove sections not greater than 75 chars
    # This is bespoke value in In Search Of for irrelevent sections

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


# Look at section chapters before and after splitting
for i, chap in enumerate(section_chapters):
    # print(f"Chapter {i-1}")
    print(map_lengths(chap))
    split_chap = map_splits("recursive", 5000, chap)
    print(map_lengths(split_chap))
    print(map_tokens(llm, chap))
    for j, sec in enumerate(split_chap):
        if len(sec) < 650:
            print(j, sec)
        else:
            print(j, sec[:200].encode("utf-8"))
    print()


def printin():
    with Session(engine) as session:
        book = Book.get(session, name=book_name)
        print(book)
        if book:
            chap = Chapter.get(session, book_id=book.id, name="CONTENTS")
            print(chap)


# printin()


def store_book_with_embeddings(fake_vector=True):
    if fake_vector:
        embeddings = FakeEmbeddings()
    else:
        embeddings = OpenAIEmbeddings()  # pyright: ignore

    with Session(engine) as session:
        book = Book.get_or_create(session, name=book_name)

        # Create all the chapters
        ordered_chapters = []
        for i, chapter in enumerate(section_chapters):
            if i == 0:  # Skip title page
                continue
            elif i == 1:
                chap = Chapter.get_or_create(session, name="CONTENTS", book_id=book.id)
            else:
                # 3rd item is Chapter 1
                chap_name = str(i - 1)
                chap = Chapter.get_or_create(session, name=chap_name, book_id=book.id)

            ordered_chapters.append(chap)

        # Create a SectionBatch so we can keep copies of old sections made with different algorithm
        # Change the tag to make new batch
        batch_tag = "SmallSecFixes"
        sec_batch = SectionBatch.get(session, tag=batch_tag)

        if not sec_batch:
            sec_batch = SectionBatch.create(session, tag=batch_tag)
            print(sec_batch)

            # Create all the text sections
            for i, chapter in enumerate(ordered_chapters):
                # Skip empty title page
                sections = section_chapters[i + 1]
                vectors = embeddings.embed_documents(sections)
                print(f"Chapter {chapter.name} embeddings fetched.")
                for j, section in enumerate(sections):
                    sec = SectionL(
                        chapter_id=chapter.id,
                        batch_id=sec_batch.id,
                        index=j,
                        text=section,
                        checksum=calculate_checksum(section),
                    )
                    session.add(sec)

            session.commit()

        # Fetch all the sections
        sections = SectionL.get_all(session, batch_id=sec_batch.id)

        # Create a chunk batch to keep copies of old chunks made with different params
        # Change the tag to save a new batch
        chunk_tag = batch_tag + "-Spacy-2000-05-26"
        chunk_batch = ChunkBatch.get(session, tag=chunk_tag)

        if not chunk_batch:
            chunk_batch = ChunkBatch.create(session, tag=chunk_tag)

            # Create all the chunks with embeddings
            for i, section in enumerate(sections):
                # Skip chapter headings, and weird small thing
                if len(section.text) < 75:
                    continue

                splits = split_doc("spacy", 2000, section.text)
                vectors = embeddings.embed_documents(splits)
                print(
                    f"Section {section.chapter_id} {section.index} embeddings fetched."
                )
                for j, split in enumerate(splits):
                    chunk = Chunk.create(
                        session,
                        batch_id=chunk_batch.id,
                        text=split,
                        checksum=calculate_checksum(split),
                        embedding=vectors[j],
                    )
                    sec_chunk = SectionChunk(
                        section_id=section.id, chunk_id=chunk.id, index=j
                    )
                    session.add(sec_chunk)

            session.commit()


# Actually fetch embeddings into db
# Do only once
# store_book_with_embeddings(fake_vector=False)

# store_book_with_embeddings()
