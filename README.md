# Building Code AI

A free, shareable web app for asking questions about building regulation PDFs.

- Drop your regulation PDFs into a `regulations/` folder once → they become a **shared library** anyone using the app can query.
- Each user can also **upload their own PDFs** in their browser session.
- Keep **multiple chats** open in parallel without re-loading the documents.
- Answers are cited with the **exact filename and page**.
- If the answer isn't in any loaded PDF, the app says so and offers a best-effort suggestion from a **web search**.
- Powered by **Google Gemini** (free tier — no credit card needed).
- Hosted free on **Streamlit Community Cloud**, shareable by URL.

---

## What you'll have at the end

A URL like `https://your-app-name.streamlit.app` that you can send to anyone. They open it in a browser and start asking questions. No installs, no logins (unless you add a password, which is optional).

---

## Setup (one time, ~15 minutes)

You'll need three free accounts: **Google AI Studio** (for the API key), **GitHub** (to host the code), and **Streamlit Community Cloud** (to host the app).

### Step 1 — Get a free Gemini API key

1. Go to https://aistudio.google.com/apikey
2. Sign in with a Google account.
3. Click **Create API key** → **Create API key in new project**.
4. Copy the key somewhere safe. It looks like `AIzaSy…`.

The free tier is 1,500 requests/day and 1M tokens/minute — plenty for a small team.

### Step 2 — Put this code on GitHub

1. Make a free account at https://github.com if you don't have one.
2. Create a new **public** repository, e.g. `building-code-ai`. (Public is required for the free Streamlit Cloud tier. Don't worry — the API key is *not* in the code; it lives in Streamlit's secrets.)
3. Upload these files to it (just drag and drop them on the GitHub web UI):
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
   - the `.streamlit/` folder (with `config.toml` and `secrets.toml.example`)
   - the `regulations/` folder (initially with just `README.txt`)

### Step 3 — Add your regulation PDFs

In the GitHub web UI, open the `regulations/` folder, click **Add file → Upload files**, and drop your PDFs in. Commit.

> Tip: Give files clean names like `Dubai-Building-Code-2021.pdf`. The filename appears as the citation users see.

You can come back and add/remove PDFs anytime — Streamlit redeploys automatically on every commit, and the app re-indexes only when the file list changes.

### Step 4 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Pick your `building-code-ai` repository, branch `main`, main file `app.py`.
4. Click **Advanced settings → Secrets** and paste:

   ```toml
   GEMINI_API_KEY = "AIzaSy…paste your key here…"
   APP_PASSWORD = ""    # leave empty for fully public; set a code for invite-only
   ```

5. Click **Deploy**.

The first deploy takes 2–5 minutes (it installs the dependencies and indexes your PDFs). You'll get a URL like `https://building-code-ai-xxxx.streamlit.app`.

Share that URL with anyone. Done.

---

## Running it locally first (optional, for testing)

If you want to try it on your own computer before deploying:

```bash
# from inside the project folder
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt

# create your secrets file
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
# then open .streamlit\secrets.toml and paste your Gemini key

streamlit run app.py
```

It opens in your browser at `http://localhost:8501`.

---

## Making the app private (optional)

If you don't want it public on the internet, set `APP_PASSWORD` in Streamlit's secrets to any passphrase. Anyone visiting the URL will be asked for the code before they can use it.

```toml
GEMINI_API_KEY = "AIzaSy…"
APP_PASSWORD = "edge-architects-2026"
```

---

## How the answers work

When a user asks a question, the app:

1. Embeds the question and finds the most relevant chunks across **all** loaded PDFs (shared library + the user's personal uploads).
2. Sends those chunks to Gemini along with strict instructions: prefer the documents, cite the filename and page, never invent regulation numbers.
3. If no chunk is relevant enough (cosine similarity below 0.55), the app does a quick DuckDuckGo web search and includes those snippets — but the model is told to mark them clearly as "from the public web, verify against your official code."

Every assistant message has a **"Show retrieved excerpts"** expander so users can see exactly which paragraphs the answer came from.

---

## Limitations to know

- **Scanned PDFs without OCR won't work.** If a PDF is just images of pages, `pypdf` extracts no text. Run it through an OCR tool (Adobe Acrobat, ABBYY, or free `ocrmypdf`) before adding it.
- **Streamlit free tier sleeps the app after inactivity.** First visit after a few hours wakes it up; takes ~30 seconds.
- **The model is not a substitute for a qualified code consultant.** It will sometimes miss nuance or context. The citations exist precisely so users can verify.
- **Personal uploads do not persist.** When a user closes the browser, their uploaded PDFs are gone. Only the shared `regulations/` folder is permanent.

---

## File overview

```
app.py                          # the entire app (~400 lines)
requirements.txt                # Python dependencies
README.md                       # this file
.gitignore                      # keep secrets and cache out of git
.streamlit/
    config.toml                 # theme + upload size limit
    secrets.toml.example        # template for your API key
regulations/
    README.txt                  # placeholder + instructions
    *.pdf                       # your regulation PDFs go here
cache/                          # auto-created; embedded chunks cached here
```
