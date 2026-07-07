# Deploying the dashboard to Hugging Face Spaces

This gives you a public, zero-install link to the dashboard. It shows bundled
reference results out of the box (`dashboard/demo_data/`) — no API keys, no
local corpus, no live compute needed on the Space itself.

Space creation is a one-time manual step through Hugging Face's web UI (no
CLI/API for that part). Updating the Space afterward is a short, repeatable
git sequence — not a single `git push`, for reasons explained below.

## One-time setup

1. Sign in at [huggingface.co](https://huggingface.co) (GitHub login works).
2. Go to **huggingface.co/new-space**.
3. Fill in:
   - **Space name**: anything, e.g. `raglens-demo`
   - **License**: `mit`
   - **Space SDK**: **Docker**. HF's "New Space" page no longer offers a
     native Streamlit card — only Gradio / Docker / Static. Once you pick
     Docker, a "Streamlit" *template* appears alongside "Blank" and others;
     pick either, it doesn't matter — the deploy below overwrites it
     completely with this repo's own `Dockerfile`.
   - **Hardware**: CPU Basic (free)
   - **Visibility**: Public (so the link is shareable)
4. Click **Create Space**. HF gives you a new git repo at
   `https://huggingface.co/spaces/<your-username>/<space-name>`, pre-populated
   with its own placeholder template files.
5. Add it as a remote:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   ```

## Deploying (first time and every update)

HF Spaces rejects plain `git push` of this repo's full history — some
binary files were committed early in this project's history (before
`.gitignore` excluded them), and HF's pre-receive hook blocks any binary
content not stored through their LFS/Xet system. The Space also doesn't
need the project's history or its non-dashboard files (notebooks, `src/`,
`experiments/`, etc.) — only `Dockerfile`, `raglens/`, `dashboard/`,
`pyproject.toml`, and `README.md`.

So each deploy pushes a single fresh commit containing just those:

```bash
git checkout --orphan hf-space-deploy
git rm -rf --cached .
git add Dockerfile README.md pyproject.toml raglens dashboard
git commit -m "Deploy raglens dashboard to Hugging Face Spaces"
git push --force space hf-space-deploy:main
git checkout -f master
git branch -D hf-space-deploy
```

Notes on this sequence:
- `git rm -rf --cached .` unstages everything the orphan checkout carried
  over, but leaves the files on disk as untracked — that's expected.
- `git checkout -f master` at the end will report switching branches
  cleanly overwriting those now-untracked files with master's tracked
  versions. This is safe: their content is identical (nothing was actually
  modified, just untracked by the `rm --cached` step) — `-f` is required
  here or the checkout refuses, seeing name collisions it doesn't know are
  harmless.
- HF reads the YAML block at the top of `README.md` (`sdk: docker`,
  `app_port: 7860`) to know it's a Docker Space, then builds and runs the
  `Dockerfile` (installs the lean base package, no Docling/ChromaDB/Ollama
  needed for the dashboard, then runs `streamlit run dashboard/app.py` on
  port 7860).

Wait a couple of minutes for the Docker image to build, then check
`https://huggingface.co/spaces/<your-username>/<space-name>`.

## What it shows

Since there's no local `experiments/` corpus or API keys on the Space, the
dashboard's `_resolve()` fallback (see `dashboard/app.py`) automatically
serves the bundled reference results in `dashboard/demo_data/` instead —
the same retrieval-benchmark and RAGAS numbers already discussed in the main
README, not live/synthetic data.

## Expectations

Free-tier Spaces sleep after a period of inactivity and wake on the next
visit (a ~30-60s cold start) — this is standard for any free hosting tier,
not something specific to this setup. If you need always-on, that's a paid
HF tier, same as Streamlit Cloud or any other host.
