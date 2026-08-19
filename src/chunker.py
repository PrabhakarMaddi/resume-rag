import re
from typing import List, Dict

# Expanded to cover real-world header variants seen across different resume templates.
# Order doesn't matter here — we sort by length before building the regex.
SECTION_HEADERS = [
    "Objective", "Career Objective",
    "Summary", "Professional Summary",
    "Education",
    "Experience", "Work Experience", "Internship Experience",
    "Professional Experience", "Internship Experience",
    "Projects",
    "Skills", "Key Skills", "Technical Skills",
    "Certifications", "Achievements", "Achievements & Certifications",
    "Achievements & Leadership",
    "Relevant Coursework",
    "Publications", "Extracurricular",
    "Coding Profiles", "Languages", "Language",
]

_SORTED_HEADERS = sorted(SECTION_HEADERS, key=len, reverse=True)

FALLBACK_MAX_CHUNKS = 1
FALLBACK_CHUNK_SIZE = 800   # characters, generous since we have no better signal
FALLBACK_OVERLAP = 100


def chunk_resume(text: str, candidate_name: str) -> List[Dict]:
    """
    Split resume text into chunks by section heading.
    Falls back to fixed-size chunking if header-based splitting fails
    (e.g. multi-column PDFs where headers land in odd positions).
    """
    chunks = _chunk_by_headers(text, candidate_name)

    if len(chunks) <= FALLBACK_MAX_CHUNKS:
        print(f"[warning] {candidate_name}: header-based chunking produced "
              f"only {len(chunks)} chunk(s) — falling back to fixed-size split.")
        chunks = _chunk_fixed_size(text, candidate_name)

    return chunks


def _chunk_by_headers(text: str, candidate_name: str) -> List[Dict]:
    pattern = r"(?=^(" + "|".join(re.escape(h) for h in _SORTED_HEADERS) + r")\s*$)"
    parts = re.split(pattern, text, flags=re.MULTILINE | re.IGNORECASE)

    chunks = []
    current_section = "Header"
    buffer = ""

    # Build a lookup so we can normalize whatever casing matched back to a clean label
    header_lookup = {h.lower(): h for h in SECTION_HEADERS}

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.lower() in header_lookup:
            if buffer.strip():
                chunks.append(_make_chunk(buffer, candidate_name, current_section))
            current_section = header_lookup[part.lower()]
            buffer = ""
        else:
            buffer += part + "\n"

    if buffer.strip():
        chunks.append(_make_chunk(buffer, candidate_name, current_section))

    return chunks


def _chunk_fixed_size(text: str, candidate_name: str,
                       chunk_size: int = FALLBACK_CHUNK_SIZE,
                       overlap: int = FALLBACK_OVERLAP) -> List[Dict]:
    """
    Fixed-size splitter with overlap, splitting on word boundaries
    so words never get cut mid-way.
    """
    words = text.strip().split()
    chunks = []

    # Convert character-based chunk_size/overlap into an approximate word count
    # (avg ~5-6 chars per word including space, adjust if needed)
    words_per_chunk = max(chunk_size // 6, 20)
    overlap_words = max(overlap // 6, 5)

    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words).strip()
        if chunk_text:
            chunks.append(_make_chunk(chunk_text, candidate_name, "Unstructured"))
        start += words_per_chunk - overlap_words

    return chunks

def _make_chunk(text: str, candidate_name: str, section: str) -> Dict:
    return {
        "text": text.strip(),
        "metadata": {
            "candidate": candidate_name,
            "section": section
        }
    }


if __name__ == "__main__":
    from loader import load_all_resumes

    resumes = load_all_resumes()
    all_chunks = []

    for name, text in resumes.items():
        chunks = chunk_resume(text, candidate_name=name)
        all_chunks.extend(chunks)
        print(f"\n{name} -> {len(chunks)} chunks")
        for c in chunks:
            print(f"  [{c['metadata']['section']}] {c['text'][:80]}...")