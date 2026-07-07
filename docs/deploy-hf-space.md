# Deploying the dashboard to Hugging Face Spaces

This gives you a public, zero-install link to the dashboard. It shows bundled
reference results out of the box (`dashboard/demo_data/`) — no API keys, no
local corpus, no live compute needed on the Space itself.

This is a one-time manual setup (Space creation happens through Hugging
Face's web UI — there's no CLI/API for that step). After that, updates are
just a normal `git push`.

## One-time setup

1. Sign in at [huggingface.co](https://huggingface.co) (GitHub login works).
2. Go to **huggingface.co/new-space**.
3. Fill in:
   - **Space name**: anything, e.g. `raglens-demo`
   - **SDK**: **Streamlit**
   - **Visibility**: Public (so the link is shareable)
4. Click **Create Space**. HF gives you a new, empty git repo at
   `https://huggingface.co/spaces/<your-username>/<space-name>`.
5. From your local clone of *this* repo:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space master:main
   ```
   (You'll be prompted for HF credentials — use an HF access token as the
   password, from huggingface.co/settings/tokens.)
6. Wait a minute or two for the Space to build. It reads the YAML block at
   the top of this repo's `README.md` (`sdk: streamlit`,
   `app_file: dashboard/app.py`) to know what to run, and installs from the
   root-level `requirements.txt` (`-e .` — the lean base install, no Docling/
   ChromaDB/Ollama needed for the dashboard).

Your public demo is now live at `https://huggingface.co/spaces/<your-username>/<space-name>`.

## What it shows

Since there's no local `experiments/` corpus or API keys on the Space, the
dashboard's `_resolve()` fallback (see `dashboard/app.py`) automatically
serves the bundled reference results in `dashboard/demo_data/` instead —
the same retrieval-benchmark and RAGAS numbers already discussed in the main
README, not live/synthetic data.

## Updating the Space later

```bash
git push space master:main
```

Same as any git remote — no need to repeat the one-time Space creation.

## Expectations

Free-tier Spaces sleep after a period of inactivity and wake on the next
visit (a ~30-60s cold start) — this is standard for any free hosting tier,
not something specific to this setup. If you need always-on, that's a paid
HF tier, same as Streamlit Cloud or any other host.
