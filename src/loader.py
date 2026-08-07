from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: Path) -> str:

    """Extract raw text from a single PDF file."""

    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def load_all_resumes(raw_dir: str = "data/raw") -> dict:

    """
    Load every PDF in the raw directory.
    Returns a dict: { filename: raw_text }

    """
    
    raw_path = Path(raw_dir)
    resumes = {}

    for pdf_file in raw_path.glob("*.pdf"):
        print(f"Loading {pdf_file.name}...")
        text = load_pdf(pdf_file)
        resumes[pdf_file.stem] = text  # stem = filename without extension

    return resumes


if __name__ == "__main__":
    resumes = load_all_resumes()
    print(f"\nLoaded {len(resumes)} resumes.\n")

    # Sanity check — print the first 500 characters of each resume
    for name, text in resumes.items():
        print(f"--- {name} ---")
        print(text[:500])
        print("...\n")