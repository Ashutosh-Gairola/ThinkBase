import os
import re
import sys
import unicodedata
import fitz                 # PyMuPDF
import docx
import pytesseract
from fuzzywuzzy import fuzz

# Detect base directory (works in source + PyInstaller EXE)
if getattr(sys, 'frozen', False):  # Running as EXE
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

TESSERACT_PATH = os.path.join(BASE_DIR, "tesseract", "tesseract.exe")
TESSDATA_PATH = os.path.join(BASE_DIR, "tesseract", "tessdata")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH

# ---------------------------------------------------------
# 1. UNIFIED DOCUMENT LOADER
# ---------------------------------------------------------
def load_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".txt":
        raw = load_text(path)

    elif ext == ".pdf":
        raw = load_pdf_with_ocr(path)

    elif ext == ".docx":
        raw = load_docx(path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    cleaned = clean_text(raw)
    cleaned = remove_headers_footers(cleaned)

    return cleaned


# ---------------------------------------------------------
# 2. TXT Loader
# ---------------------------------------------------------
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---------------------------------------------------------
# 3. DOCX Loader
# ---------------------------------------------------------
def load_docx(path: str) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


# ---------------------------------------------------------
# 4. PDF LOADER WITH OCR + FUZZY CLEANING
# ---------------------------------------------------------
def load_pdf_with_ocr(path: str) -> str:
    doc = fitz.open(path)
    extracted_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text()

        if text.strip():
            # Normal PDF (text-based)
            extracted_text.append(text)
        else:
            # Likely SCANNED PDF — use OCR
            pix = page.get_pixmap(dpi=200)  # Higher DPI improves OCR
            ocr_image = pix.pil_tobytes(format="PNG")

            ocr_text = pytesseract.image_to_string(ocr_image)

            # Optional fuzzy-based cleanup
            ocr_text = fuzzy_cleanup(ocr_text)

            extracted_text.append(ocr_text)

    return "\n".join(extracted_text)


# ---------------------------------------------------------
# 4A. Fuzzy Cleanup For OCR Text
# ---------------------------------------------------------
def fuzzy_cleanup(text: str) -> str:
    """
    Removes OCR garbage:
    - random symbols
    - lines with low alphanumeric ratio
    - lines too similar to noise patterns
    """

    clean_lines = []
    for line in text.splitlines():

        # Remove lines that are mostly symbols
        if fuzz.ratio(line, re.sub(r"\W+", "", line)) < 40:
            # if similarity is low, it's mostly symbols
            continue

        # Drop extremely short garbage
        if len(line.strip()) < 3 and not line.isdigit():
            continue

        # Strip noise
        line = re.sub(r"[^A-Za-z0-9.,;:?!'\"]+", " ", line)

        clean_lines.append(line.strip())

    return "\n".join(clean_lines)


# ---------------------------------------------------------
# 5. HEADER/FOOTER REMOVAL BASED ON PATTERNS & REPEATED LINES
# ---------------------------------------------------------
def remove_headers_footers(text: str) -> str:
    lines = text.splitlines()

    # --- Detect repeated lines (likely headers/footers) ---
    line_counts = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line_counts[line] = line_counts.get(line, 0) + 1

    # Lines repeated more than 3× across document = header or footer
    repeated_lines = {line for line, count in line_counts.items() if count >= 3}

    cleaned = []
    for line in lines:
        stripped = line.strip()

        if stripped in repeated_lines:
            continue

        # Page numbers:  "Page 3", "3/10", "- 4 -"
        if re.match(r"^\s*(Page\s*\d+|\d+\s*/\s*\d+|-+\s*\d+\s*-+)\s*$", stripped, re.IGNORECASE):
            continue

        # Company names repeated? fuzzy detect (optional)
        for rep in repeated_lines:
            if fuzz.ratio(stripped, rep) > 80:
                break
        else:
            cleaned.append(line)

    return "\n".join(cleaned)


# ---------------------------------------------------------
# 6. CLEANING PIPELINE
# ---------------------------------------------------------
def clean_text(text: str) -> str:
    """
    General noise cleaning:
    - unicode normalization
    - remove multiple blank lines
    - remove symbol-only lines
    - fix hyphenated line breaks
    - heavy whitespace normalization
    """

    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove non-printable chars
    text = "".join(ch for ch in text if ch.isprintable() or ch.isspace())

    # Remove hyphen line-breaks: "com-\nputer" -> "computer"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Remove symbol-only lines
    text = "\n".join(
        line for line in text.splitlines()
        if not re.fullmatch(r"[\W_]{4,}", line.strip())
    )

    # Collapse multiple newlines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Strip whitespace
    text = "\n".join(line.strip() for line in text.splitlines())

    # Normalize spaces
    text = re.sub(r" +", " ", text)

    return text.strip()


# ---------------------------------------------------------
# 7. CHUNKING
# ---------------------------------------------------------
def chunk_text(text: str, size: int = 300):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]


# ---------------------------------------------------------
# 8. DOC ID
# ---------------------------------------------------------
def sanitize_doc_id(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    return name.replace(" ", "_").lower()
