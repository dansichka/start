# CiNii MCP Server

This module exposes a small [FastAPI](https://fastapi.tiangolo.com/) service that acts as an MCP server for the academic database [CiNii](https://ci.nii.ac.jp/).

## Setup

1. Create a Python virtual environment and install the requirements:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```

2. Obtain an API key from CiNii and export it as `CII_APPID`:
   ```bash
   export CII_APPID=your_api_key
   ```

3. Add the `src` directory to `PYTHONPATH` and run the service with `uvicorn`:
   ```bash
   export PYTHONPATH=./src  # on Windows use: set PYTHONPATH=src
   uvicorn cinii_mcp_server.main:app --reload
   ```

The API documentation will be available at `http://localhost:8000/docs`.
