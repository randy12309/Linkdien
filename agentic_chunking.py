"""Agentic document chunking using Google Gemini for semantic boundary detection."""
import os
import json
import re
from typing import Iterator, List
from pydantic import BaseModel, Field

# Import google-genai
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-3.1-flash-lite")

# Maximum characters per chunk for embedding
CHUNK_MAX_CHARS = 1500
CHUNK_OVERLAP = 200


class ChunkPlan(BaseModel):
    """Represents the chunking plan returned by the agent."""
    chunks: List[str] = Field(description="List of chunk texts")


def _build_chunking_prompt(text: str, context: str = "", max_chars: int = CHUNK_MAX_CHARS) -> str:
    """Build the prompt for agentic chunking; `context` is the tail of what was already chunked from earlier pages."""
    previous = f"""
PREVIOUS CONTEXT (the last chunks already produced from the earlier pages; for reference only, never repeat it in your output):
{context}

If the new text continues a topic from the previous context, keep that topic's wording consistent and start the first chunk with the heading/topic it belongs to.
""" if context else ""
    return f"""You are an expert document processor. Your task is to split the following document text into semantically coherent chunks for vector embedding.
{previous}
RULES:
1. Each chunk should be a complete, self-contained semantic unit (a topic, section, or logical group of paragraphs)
2. Target chunk size: ~{max_chars} characters (range: {max_chars//2} to {max_chars*2})
3. Preserve complete sentences and paragraphs - never cut mid-sentence
4. Group related content together (e.g., a policy definition with its examples, a FAQ with its answer)
5. Keep headers/titles with their content
6. Cover ALL of the text below and do not invent content
7. Return ONLY a JSON object with a "chunks" array of strings

DOCUMENT TEXT:
{text}

Return format:
{{
  "chunks": [
    "First chunk text...",
    "Second chunk text...",
    ...
  ]
}}"""


def agentic_chunk(text: str, context: str = "", max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    """
    Use Gemini to intelligently chunk a document into semantic units.

    Falls back to simple chunking if Gemini is unavailable or fails.
    """
    if not GENAI_AVAILABLE:
        print("google-genai not available, falling back to simple chunking")
        return simple_chunk(text, max_chars)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set, falling back to simple chunking")
        return simple_chunk(text, max_chars)

    try:
        client = genai.Client(api_key=api_key)

        prompt = _build_chunking_prompt(text, context, max_chars)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        if response.text:
            result = json.loads(response.text)
            chunks = result.get("chunks", [])

            # Validate and clean chunks
            cleaned_chunks = []
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk and len(chunk) > 50:  # Minimum meaningful chunk size
                    # Ensure chunk doesn't exceed max significantly
                    if len(chunk) > max_chars * 2:
                        # Split oversized chunks
                        cleaned_chunks.extend(simple_chunk(chunk, max_chars))
                    else:
                        cleaned_chunks.append(chunk)

            if cleaned_chunks:
                return cleaned_chunks

    except Exception as e:
        print(f"Agentic chunking failed: {e}, falling back to simple chunking")

    return simple_chunk(text, max_chars)


def simple_chunk(text: str, size: int = 1000, overlap: int = 150) -> List[str]:
    """Fallback: Pack paragraphs into chunks of about `size` characters."""
    import re
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        while len(paragraph) > size:
            cut = paragraph.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            units.append(paragraph[:cut])
            paragraph = paragraph[cut:].lstrip()
        if paragraph:
            units.append(paragraph)
    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > size:
            chunks.append(current)
            tail = current[-overlap:]
            current = tail[tail.find(" ") + 1:] if " " in tail else tail
        current = f"{current} {unit}".strip()
    if current:
        chunks.append(current)
    return chunks


PAGES_PER_CALL = 2
CONTEXT_CHARS = 1500


def iter_chunk_batches(pages: List[str], pages_per_call: int = PAGES_PER_CALL) -> Iterator[List[str]]:
    """Chunk a document two pages at a time, yielding each window's chunks so the caller can store them before the
    next window is processed. The tail of the chunks made so far is kept in memory and passed to the model with
    the next window, so a topic that runs across a page boundary is split consistently."""
    previous: List[str] = []
    for start in range(0, len(pages), pages_per_call):
        window = "\n\n".join(pages[start:start + pages_per_call])
        batch = agentic_chunk(window, "\n\n".join(previous[-2:])[-CONTEXT_CHARS:])
        previous = batch or previous
        yield batch
