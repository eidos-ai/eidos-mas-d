# Eidos MCP Servers

This project provides a set of Model-Context-Protocol (MCP) servers that expose various tools.

## Features

*   **Research Server**: Find papers on arXiv, extract information, and interact with a research-focused chatbot.
*   **Pizza Ordering Server**: Analyze pizza order history and get recommendations for future orders.
*   **Find Server**: A utility to find files on your filesystem.
*   **MCP Compliant**: Interact with the servers using any MCP-compatible client.
*   **Easy to Inspect**: Visualize and interact with the servers using `@modelcontextprotocol/inspector`.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Python 3.12+
*   [uv](https://github.com/astral-sh/uv) - An extremely fast Python package installer and resolver.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/eidos-project/eidos-mcp-servers.git
    cd eidos-mcp-servers
    ```

2.  **Create a virtual environment and install dependencies:**
    This project uses `uv` for package management.
    ```sh
    uv venv
    source .venv/bin/activate
    uv sync
    ```
    This will install all the dependencies listed in `pyproject.toml`.

### Configuration

1.  Create a `.env` file in the root of the project.
2.  Add a `DATA_DIR` variable to specify where to store required information for the `research_server` and `pizza_ordering_server`:
    ```
    DATA_DIR=./data
    ```
3.  If you intend to run the chatbot scripts in `research_server`, you must also provide `ANTHROPIC_API_KEY`:
    ```
    ANTHROPIC_API_KEY="your_api_key_here"
    ```

## Usage

This project contains multiple MCP servers. You can run each one and inspect it using the MCP Inspector.

### Research Server

The `research_server` provides tools to search for scientific papers on arXiv.

**Running the Server:**
```sh
uv run python research_server/search_papers_mcp_server.py
```

**Inspecting with MCP Inspector:**
```sh
npx @modelcontextprotocol/inspector uv run python research_server/search_papers_mcp_server.py
```

### Pizza Ordering Server

The `pizza_ordering_server` provides tools to analyze pizza ordering history. It needs a `pizzas_dataset.csv` inside a `data/pizza` directory.

**Running the Server:**
```sh
uv run python pizza_ordering_server/pizza_ordering_server.py
```

**Inspecting with MCP Inspector:**
```sh
npx @modelcontextprotocol/inspector uv run python pizza_ordering_server/pizza_ordering_server.py
```

### Find Server

The `find_server` provides a tool to find files on your local file system.

**Running the Server:**
```sh
uv run python find_server/find_server.py
```

**Inspecting with MCP Inspector:**
```sh
npx @modelcontextprotocol/inspector uv run python find_server/find_server.py
```

## Project Structure

```
.
├── .venv/
├── data/
│   ├── papers/
│   └── pizza/
├── find_server/
│   └── find_server.py
├── pizza_ordering_server/
│   └── pizza_ordering_server.py
├── research_server/
│   ├── chatbot.py
│   ├── mcp_chatbot.py
│   ├── search_papers.py
│   ├── search_papers_mcp_client.py
│   └── search_papers_mcp_server.py
├── .env
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```
