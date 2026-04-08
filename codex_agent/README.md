# 🤖 Codex Agent Environment

Welcome to the **Codex Agent Environment** for AutoResearch-RL! 🎉

This directory contains everything you need to unleash OpenAI's Codex (or a local equivalent) on our perpetual codebase. AutoResearch-RL is a beast of a pipeline designed to mutate code and hunt for the next State-of-the-Art (SOTA) model. To do this autonomously, we need to give our LLM overlords a comfortable, well-lit sandbox to play in. That's exactly what this folder is!

## 🌟 What's in the Box?

We've packed this directory with context, scripts, and configurations so your agent can hit the ground running:

- **📜 `AGENTS.md`**: The holy grail of context. It tells the agent exactly what the project is, how it's structured, what the draconian constraints are (16MB size limit, 10 min train time), and what success looks like.
- **🚀 `run_autoresearch.sh`**: The master "do everything" button. It installs dependencies, runs tests, loads environment variables, and launches the perpetual research loop (`main.py`). The agent just needs to push this button.
- **🧪 `ci.sh`**: A quick-and-dirty test runner. The agent can use this to make sure it hasn't broken anything before launching a full GPU training run.
- **⚙️ `.env.example`**: A template for environment variables (like `MAX_ITERATIONS` and `AUTORESEARCH_MODE`). Copy this to `.env` if you want to override the defaults.
- **📦 `requirements.txt`**: A locked-down list of Python dependencies so the agent doesn't have to guess what `numpy` or `torch` versions to install.
- **🧠 `.codex/config.toml`**: Configurations specifically for the Codex CLI tool, telling it how hard to think and whether to use local or remote models.

## 🚀 How to Unleash the Agent

If you're a human guiding an agent (or an agent reading this, hello!), here is your workflow:

1.  **Set the Environment**: Copy `.env.example` to `.env` and fill in any necessary keys (like `OPENAI_API_KEY` if you are using the cloud, or leave it blank/commented for local OSS mode).
2.  **Understand the Mission**: Read `AGENTS.md` to internalize the goals and constraints of the Parameter-Golf challenge.
3.  **Run the Pipeline**: Execute `./run_autoresearch.sh`. Watch the orchestrator, auditor, and SPRT filter do their magic as the system loops and learns!
4.  **Iterate and Mutate**: Review the logs in `experiment_logs.jsonl` in the root folder. Did the BPB go down? Great! Did it crash? The logs will tell you why (e.g., causality leak, syntax error). Propose a new patch and try again!

## 🛠️ Operating Modes

-   **Cloud/OpenAI Mode**: Provide an `OPENAI_API_KEY`. The PPO agent will use standard OpenAI models to generate code patches.
-   **Local/OSS Mode**: Configure `.codex/config.toml` to point to a local Ollama/vLLM endpoint. Perfect for off-grid, free, continuous experimentation!

---
*Happy Hunting! Let's break some compression records.* 📉✨