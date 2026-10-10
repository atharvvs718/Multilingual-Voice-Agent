"""Read the text of every page of a PDF, in a process of its own.

    python -m rag.pdf_text book.pdf      # prints {"pages": [[1, "text"], ...]} or {"error": "..."}

PDFium is native code and is not safe to call from several threads, which is how a web
server calls things: inside the app it crashed the whole process on a second upload. Run
separately, the worst a bad PDF can do is make this one process fail.
"""
import json
import re
import shutil
import sys

from agent.lang import devanagari_ratio

# Old-font PDFs extract as Latin letters with lots of accents and symbols instead of Devanagari.
_LEGACY_FONT_JUNK = re.compile(r"[¤-¿À-ÿ]")
CANNOT_OPEN = "This PDF could not be opened. It may be damaged or need a password."
NEEDS_OCR = (
    "No readable text in this PDF. It looks scanned or uses an old font; "
    "reading it needs Tesseract OCR, which is not installed."
)


def looks_garbled(text: str) -> bool:
    """True when a page that should be Hindi/Marathi came out of the PDF as font garbage."""
    letters = sum(c.isalpha() for c in text)
    if letters < 50:
        return True                       # scanned page: no text layer at all
    junk = len(_LEGACY_FONT_JUNK.findall(text)) / letters
    return junk > 0.05 and devanagari_ratio(text) < 0.2


def extract(path: str) -> dict:
    """{"pages": [[position in the file, text], ...]} for the readable pages, or {"error": message}.

    PDFium opens "no copying" PDFs, which many textbooks are. Without OCR a blank, scanned
    or old-font page is skipped, not fatal: one picture page should not block a whole chapter.
    """
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(path)
    except pdfium.PdfiumError:
        return {"error": CANNOT_OPEN}
    can_ocr = shutil.which("tesseract") is not None
    pages, unreadable = [], 0
    for index in range(len(pdf)):
        page = pdf[index]
        textpage = page.get_textpage()
        text = textpage.get_text_bounded()
        textpage.close()
        if looks_garbled(text):
            if can_ocr:
                import pytesseract  # needs the hin and mar language packs

                text = pytesseract.image_to_string(page.render(scale=300 / 72).to_pil(), lang="mar+hin+eng")
            else:
                text = None
                unreadable += 1
        page.close()
        if text is not None:
            pages.append([index + 1, text])
    pdf.close()
    if not pages and unreadable:
        return {"error": NEEDS_OCR}
    return {"pages": pages}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    json.dump(extract(sys.argv[1]), sys.stdout, ensure_ascii=False)
