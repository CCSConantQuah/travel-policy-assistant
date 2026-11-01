# AI Travel Policy Compliance Assistant

An individual AI Bootcamp capstone proof of concept for officers planning official travel.

## Phase 1 scope

Implemented: demonstration login, sidebar navigation, Home, Travel Policy Q&A preview, Compliance Checker preview, disabled Document Upload, About Us, and Methodology placeholders.

No OpenAI calls, RAG, embeddings, ChromaDB, PDF ingestion, or compliance logic are implemented. The preview buttons are disabled and no policy decisions are generated. Use synthetic travel policy documents only; do not submit internal agency documents to external services.

## Project structure

```text
travel-policy-assistant/
├── app.py                       # Streamlit entry point and navigation
├── travel_assistant/
│   ├── __init__.py               # Python package marker
│   ├── auth.py                   # Settings and session-based demo login
│   └── pages.py                  # Six page rendering functions
├── requirements.txt             # Phase 1 dependencies
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

Streamlit provides the browser UI, widgets, and session state. `python-dotenv` loads a local `.env` file. AI-related dependencies will be added only when their phase begins.

## Configure demo login

```bash
cp .env.example .env
```

Edit `.env` and replace both values with your own demo username and password. Do not use real organizational credentials. No API key is needed in Phase 1.

Alternatively create `.streamlit/secrets.toml` locally:

```toml
DEMO_USERNAME = "your-demo-username"
DEMO_PASSWORD = "your-demo-password"
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
5. Enter sample Q&A and travel details. Confirm action buttons remain disabled and no answers or compliance decisions appear.
6. Visit Document Upload; confirm the uploader is disabled and the synthetic-document warning is visible.
7. Review About Us and all three Methodology placeholder sections.
8. Log out; confirm navigation disappears and the login form returns. Sign in again to confirm the flow still works.
9. Run `git status --short --untracked-files=all`; confirm `.env`, `.streamlit/secrets.toml`, and `.venv/` are absent from files to commit.

Optional syntax check:

```bash
python -m compileall app.py travel_assistant
```

## Git

This folder already has Git initialized; do not reinitialize it or recreate the virtual environment. Inspect and commit the Phase 1 files:

```bash
git status
git add app.py travel_assistant requirements.txt .gitignore .env.example README.md
git diff --cached
git commit -m "feat: add Phase 1 Streamlit scaffold with demo login and navigation"
```

If Git asks for your identity, configure it for this repository and retry the commit:

```bash
git config user.name "Your Name"
git config user.email "your-email@example.com"
```

For a different folder without Git, run `git init` once before the add/commit commands. A GitHub remote is optional for local commits. Always inspect staged changes for credentials before committing.

## Next phases

Later work may implement synthetic PDF processing, retrieval, grounded Q&A, and compliance assessment with citations. Phase 2 starts only on explicit request.
