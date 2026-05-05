Place your building regulation PDFs in this folder.

Every PDF in this folder becomes part of the SHARED LIBRARY — meaning
every user of the deployed app will be able to ask questions against
these documents.

Supported: any text-based PDF (scanned PDFs without OCR will not work
well; run them through an OCR tool first).

Tips:
- Keep filenames descriptive, e.g. "Dubai-Building-Code-2021.pdf".
  The filename is shown to users as the source citation.
- After adding/removing PDFs, push the change to GitHub. Streamlit
  Cloud will redeploy automatically and re-index the library on first
  load (this can take a few minutes for large PDFs).
- The first index is cached to ./cache/ so subsequent restarts are fast.
