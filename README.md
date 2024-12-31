# Assistants

A tool to allow developers to craft powerful AI assistants that can perform a variety of tasks.

Based on YouTube training <https://www.youtube.com/watch?v=qHPonmSX4Ms>

## Assistants API

Assistants API extends the existing OpenAI API... Easier to build AI assistants AS Bots, AI tools, etc...

Assistants API - What Problem It Solves?

## Assistants API Benefits

Building Complex Al Applications is very difficult!

Developers need to:

- Manage infrastructures
- Data
- Models
- Prompts
- Application state
- Embeddings,
- Storage mechanism
- ...

Developers need to spend most of their time stitching tech together, instead of actually solving customer problems.

Assistants API helps with:

- Persistent threading for ongoing conversations: Being able to save messages & context of the conversation
- Retrieval mechanisms for digging through data: Upload files for the models to use for additional knowledge-base
- Code Interpreter: Write, analyze code...
- Function calling to execute custom tasks with ease

To manually create an assistant (or agent) go to: <https://platform.openai.com/playground/assistants>

Get the API key from <https://platform.openai.com/settings/organization/api-keys>

## News API

Go to <https://newsapi.org> to get an API key.

## Python Streamlit App

Create a fresh virtual environment for better code/deployment handling.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip install -r requirements.txt
pip list -v
```

Create a `.env` file with the following content:

```sh
OPENAI_API_KEY="sk-..."
NEWS_API_KEY="..."
```

Run the Python app:

```sh
streamlit run main.py
```
