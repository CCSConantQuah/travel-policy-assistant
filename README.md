# AI Travel Policy Compliance Assistant

An individual AI Bootcamp capstone proof of concept for officers planning official travel.

## Current scope: Phase 2

Implemented: demonstration login, sidebar navigation, Home, Q&A and Compliance Checker previews, synthetic PDF ingestion, page-aware text chunking, OpenAI embeddings, persistent local Chroma storage, development retrieval, About Us, and Methodology placeholders.

No chatbot, generated answers, or compliance logic is implemented. Q&A and compliance buttons remain disabled. Use synthetic travel policy documents only; do not submit internal agency documents to external services. Processing sends extracted policy text to OpenAI; development retrieval sends the search query for embedding. Only the embeddings endpoint is used.

## Project structure

```text
travel-policy-assistant/
├── app.py                       # Streamlit entry point and navigation
├── travel_assistant/
│   ├── __init__.py               # Python package marker
│   ├── auth.py                   # Settings and session-based demo login
│   ├── pages.py                  # Existing pages and navigation mapping
│   ├── documents.py              # PDF extraction and page-aware chunking
│   ├── vector_store.py           # Embeddings, Chroma, safe replacement, retrieval
│   └── upload_page.py            # Upload/status/development-search interface
├── tests/test_ingestion.py       # Synthetic fixtures and mocked API tests
├── scripts/create_synthetic_policy.py # Creates a fictional test PDF
├── requirements.txt             # App dependencies
├── requirements-dev.txt         # App dependencies plus pytest
├── .env.example                 # Credential configuration template
├── .gitignore                   # Excludes credentials and local files
└── README.md
```

Your existing `.git/` and `.venv/` are preserved. `.env` is created locally by you and ignored by Git.

`app.py` loads local environment variables, configures the page, calls `require_login()`, and renders the page selected in the sidebar. `get_setting()` reads environment variables first, then top-level Streamlit secrets. `require_login()` checks credentials and stores the login state in the current Streamlit session. `PAGES` maps sidebar labels to small page functions, making future changes easy to locate.

## Windows setup using Git Bash

Open Git Bash in this project folder. Your virtual environment already exists, so activate it:

```bash
source .venv/Scripts/activate
python --version
python -m pip install -r requirements.txt
```

For a fresh clone only, create the virtual environment before activation:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

Streamlit provides the UI and session state. `python-dotenv` loads `.env`; PyMuPDF extracts PDF text; `langchain-text-splitters` provides LangChain's recursive splitter; `openai` calls the embeddings API; `chromadb` stores and searches vectors locally. `requirements-dev.txt` additionally installs pytest. The full LangChain framework and chat-model integrations are unnecessary in this phase.

## Configure demo login

```bash
test -f .env || cp .env.example .env
```

If `.env` already exists, preserve it rather than copying over it. Set your demo username and password and add your own `OPENAI_API_KEY` locally. Do not use real organizational login credentials. Login works without an API key; ingestion and development retrieval require it.

```dotenv
OPENAI_API_KEY=your-own-api-key
```

Never print or commit the actual key. Restart the app after changing environment configuration.

Alternatively create `.streamlit/secrets.toml` locally:

```toml
DEMO_USERNAME = "your-demo-username"
DEMO_PASSWORD = "your-demo-password"
OPENAI_API_KEY = "your-own-api-key"
```

Both credential files are ignored by Git. Environment values take precedence over Streamlit secrets. Restart the application after changing credentials. Missing or empty credentials prevent login and show a setup message.

This is demonstration authentication only: it uses a single shared account, with no identity provider, rate limiting, or production access controls. Login lasts within a Streamlit session. Log out using the sidebar.

## Run

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Stop with **Ctrl+C**. Run `deactivate` when finished using the environment.

## Manual acceptance tests

1. Before configuring credentials, start the app and confirm it displays the configuration error and no page navigation.
2. Configure credentials and restart. Submit incorrect or empty credentials; confirm access is denied with a generic error.
3. Sign in with your configured credentials; confirm Home and the sidebar appear.
4. Visit all six pages and confirm each title and description matches its navigation label.
5. Enter sample Q&A and travel details. Confirm their action buttons remain disabled and no answers or compliance decisions appear.
6. Visit Document Upload; confirm the synthetic-document warning and confirmation checkbox are visible. Follow the Phase 2 upload checks below.
7. Review About Us and all three Methodology placeholder sections.
8. Log out; confirm navigation disappears and the login form returns. Sign in again to confirm the flow still works.
9. Run `git status --short --untracked-files=all`; confirm `.env`, `.streamlit/secrets.toml`, and `.venv/` are absent from files to commit.

Optional syntax check:

```bash
python -m compileall app.py travel_assistant
```

## Git

This folder already has Git initialized; do not reinitialize it or recreate the virtual environment. Inspect and commit the Phase 2 files:

```bash
git status
git add app.py travel_assistant tests scripts requirements.txt requirements-dev.txt .gitignore .env.example README.md
git diff --cached
git commit -m "feat: add policy PDF ingestion and persistent semantic retrieval"
```

If Git asks for your identity, configure it for this repository and retry the commit:

```bash
git config user.name "Your Name"
git config user.email "your-email@example.com"
```

For a different folder without Git, run `git init` once before the add/commit commands. A GitHub remote is optional for local commits. Always inspect staged changes for credentials before committing.

## Phase 2 ingestion and storage

`travel_assistant/documents.py` contains two local operations:

- `extract_pdf()` opens uploaded bytes using PyMuPDF, extracts selectable text per page, and returns LangChain Documents plus the total PDF page count. Metadata contains the original base filename (`source`) and physical, one-based PDF page number (`page`), including gaps for blank pages. Password-protected, damaged, oversized, and textless PDFs are rejected. PDFs are limited to 20 MB and 300 pages.
- `chunk_pages()` uses `RecursiveCharacterTextSplitter` with a 1,000-character maximum and up to 200 characters of overlap. Each page is split independently so every chunk has an unambiguous source page. Chunks also receive a zero-based `chunk_index`. Overlap depends on natural separators and is not guaranteed to be exactly 200 characters.

`travel_assistant/vector_store.py` provides:

- `embed_texts()`: calls OpenAI's `text-embedding-3-small` in batches of up to 64 inputs. It checks response indices, vector dimensions, and finite values. API errors are replaced with safe messages, without displaying provider exception details or keys. Requests have a 60-second timeout and one SDK retry.
- `process_policy()`: extracts, chunks, embeds, and writes a uniquely named staging Chroma collection with cosine distance. It verifies the stored chunk count before publishing. Chroma persists locally without a separate server or explicit persist call.
- `_publish()`: writes and flushes a temporary JSON manifest, then uses `os.replace()` to atomically switch `active_policy.json`. Until that switch succeeds, the old policy remains active. Failure during embeddings, storage, or publication does not remove the previous collection.
- `get_active_policy()`: reads the manifest from disk and verifies the collection exists and its chunk count matches. Status survives app restarts.
- `retrieve_policy(query, api_key)`: embeds a development query and returns the top three nearest chunks (fewer when the active policy has fewer), including text, filename, page, chunk index, and cosine distance. It generates no answer.

Storage lives under `data/chroma_db/`, resolved relative to the project rather than the shell directory. It includes extracted text, vectors, metadata, Chroma files, and the active-policy manifest. The original uploaded PDF is handled in memory and is not saved by ingestion. `data/`, PDFs, local databases, secrets, and the environment are excluded from Git.

After successful publication the previous collection is deleted. If cleanup fails, the UI warns; the new policy remains active and old data is never searched. Interrupted uploads can leave inactive staging collections on disk, but retrieval only opens the manifest's active collection. This is logical replacement, not forensic secure erasure.

`travel_assistant/upload_page.py` handles the uploader, synthetic-data confirmation, progress spinner, success/failure feedback, active-policy metrics, and development retrieval panel. The existing navigation still maps to the Document Upload page; login and other page interfaces are preserved.

## Test upload and semantic retrieval

Activate the existing virtual environment and update dependencies:

```bash
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python scripts/create_synthetic_policy.py
python -m streamlit run app.py
```

1. Configure your own `OPENAI_API_KEY` in `.env` or top-level Streamlit secrets, then restart. Do not share the key.
2. Sign in, open Document Upload, and select `data/synthetic_policy.pdf`. This generated file is entirely fictional and ignored by Git.
3. Confirm it is synthetic, then select **Process policy**. A success message should appear only after embedding and storage finish. The generated sample has three pages and three chunks; status should be Ready.
4. Expand **Development only: test semantic retrieval**. Search for `What hotel rate is permitted?`. Inspect the top passages and source pages; the hotel passage is on physical PDF page 2. Semantic ranking may vary; distance is not a confidence or compliance score.
5. Restart the app and sign in again. Confirm the active-policy metrics and retrieval still work.
6. Process a different fictional policy. Confirm the filename and counts update and retrieval contains only the replacement document's sources.
7. Try a damaged PDF or an image-only PDF. Confirm a helpful error and that the active policy remains available. For an API failure check, temporarily remove or change your key and restart; confirm processing fails without replacing the old policy. Restore the correct key afterwards.
8. A PDF with some blank/image-only pages is accepted if other pages have text; the UI warns about skipped pages. Review extracted results before using any later feature.

Each explicit processing action incurs embedding requests for the document again; normal navigation/reruns do not. Each submitted development search embeds the query. The application does not call a chat/completions API. Do not use a real agency document even for a test.

## Automated tests (no API key or paid requests)

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests create fictional PDFs in memory and use temporary local Chroma directories. They mock external embeddings calls and exercise extraction, physical page metadata, chunking and overlap, retrieval ranking and the three-result limit, persistence metadata, replacement isolation, failed embeddings/storage/publication, cleanup failures, missing keys, and API response batching. No `.env` credentials are loaded by these tests.

## Limitations and security

- This is a local, single-process proof of concept with one shared active policy. A process-wide lock serializes ingestion and retrieval across Streamlit sessions; multiple app processes pointing at the same database are unsupported.
- Demo login is not production authentication. Local stored text and vectors are not encrypted by this app. All logged-in users share the active document.
- No OCR, table reconstruction, or layout reasoning is implemented. Complex columns, tables, headings, or mixed scanned/text pages may extract imperfectly. Physical page numbers may differ from printed page labels. Page boundaries can split a policy rule.
- Selecting a PDF uploads it to the Streamlit process. External embedding calls start only after the user confirms and clicks Process policy. The code does not log keys or raw API errors. Chroma anonymized telemetry is disabled.
- Embedding requests may incur API charges. The character-based limits and batches keep the proof of concept manageable, but are not a complete cost-control or tokenizer system.
- Failed uploads preserve the active policy, but are not automatically resumable. A crash can leave inactive disk data; stop the app before manually backing up or cleaning the data folder. Back up the entire directory together, including the manifest. A corrupt active record/database is reported rather than silently overwritten.
- The automated tests do not verify live OpenAI account access, quota, or semantic quality. Verify one synthetic upload with your own key after setup.

References: [OpenAI embeddings API](https://developers.openai.com/api/reference/python/resources/embeddings/methods/create), [LangChain recursive splitting](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter), [Chroma collection configuration](https://docs.trychroma.com/docs/collections/configure), [PyMuPDF text extraction](https://pymupdf.readthedocs.io/en/latest/recipes-text.html).

## Next phases

Grounded Q&A and compliance assessment with citations remain future work. Phase 3 starts only on explicit request.
