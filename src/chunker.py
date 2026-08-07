import re
from typing import List, Dict

# Common resume section headers to split on
SECTION_HEADERS = [
    "Objective", "Summary", "Education", "Experience",
    "Work Experience", "Projects", "Skills", "Technical Skills",
    "Certifications", "Achievements", "Publications", "Extracurricular"
]


def chunk_resume(text: str, candidate_name: str) -> List[Dict]:
    """
    Split resume text into chunks by section heading.
    Returns a list of dicts: { text, metadata }
    """
    # Build a regex that matches any of our known headers at the start of a line
    pattern = r"(?=^(" + "|".join(SECTION_HEADERS) + r")\s*$)"
    parts = re.split(pattern, text, flags=re.MULTILINE)

    chunks = []
    current_section = "Header"  # anything before the first matched heading
    buffer = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in SECTION_HEADERS:
            # save previous section before starting a new one
            if buffer.strip():
                chunks.append(_make_chunk(buffer, candidate_name, current_section))
            current_section = part
            buffer = ""
        else:
            buffer += part + "\n"

    # save the last section
    if buffer.strip():
        chunks.append(_make_chunk(buffer, candidate_name, current_section))

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