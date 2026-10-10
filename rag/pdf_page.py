"""Render pages of a PDF as JPEG pictures, in a process of its own.

PDFium is not safe to call from the server's threads (it crashed the app when it was), so
this runs the same way rag.pdf_text does:

    python -m rag.pdf_page book.pdf out_folder 21 22

Each page given (positions in the file, from 1) is written as out_folder/page_<n>.jpg.
"""
import sys
from pathlib import Path

import pypdfium2 as pdfium

SCALE = 1.6         # about 950 x 1270 pixels for a textbook page: small print and equations stay readable
QUALITY = 80


def render(pdf_path: str, out_folder: str, positions: list[int]) -> list[str]:
    """Write one JPEG per page position that exists; returns the paths written."""
    document = pdfium.PdfDocument(pdf_path)
    written = []
    for position in positions:
        if 1 <= position <= len(document):
            path = Path(out_folder) / f"page_{position}.jpg"
            document[position - 1].render(scale=SCALE).to_pil().convert("RGB").save(path, format="JPEG", quality=QUALITY)
            written.append(str(path))
    return written


if __name__ == "__main__":
    for written_path in render(sys.argv[1], sys.argv[2], [int(position) for position in sys.argv[3:]]):
        print(written_path)
