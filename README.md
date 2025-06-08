# Agent Zero Memory Alternatives

This repository provides a sample `docker-compose.yml` for running
`agent-zero` with the default `mcp/memory` server.  The memory container
communicates with Agent Zero via standard input and stores data in the
Docker volume `claude-memory`.

The examples below show how to expose the agent's memory to other
systems using free, open‑source services like DuckDB, Qdrant, Neo4j, or
a simple HTTP server. Each scenario includes a dedicated compose file,
health checks, and environment variables required by Agent Zero.

---

## 1. Using SQLite for DuckDB access

`mcp/sqlite` stores the knowledge base in a single SQLite file.  DuckDB
can read this file directly via its `sqlite_scanner` extension.

Run with:

```bash
docker-compose -f docker-compose.duckdb.yml up -d
```

### docker-compose.duckdb.yml
```yaml
version: '3.9'

volumes:
  memory-sqlite:

networks:
  a0net: {}

services:
  memory:
    image: mcp/sqlite:latest
    container_name: memory-sqlite
    stdin_open: true
    volumes:
      - memory-sqlite:/data
    networks:
      - a0net

  duckdb:
    image: datacatering/duckdb:v1.3.0
    container_name: duckdb
    command: ["duckdb", "--listen", "0.0.0.0:8000"]
    volumes:
      - memory-sqlite:/data:ro
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "duckdb", "--version"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - a0net

  agent-zero:
    build: ./agent-zero
    container_name: agent-zero
    volumes:
      - memory-sqlite:/data
    depends_on:
      - memory
    ports:
      - "50080:80"
    networks:
      - a0net
```

DuckDB can now query the SQLite file in `memory-sqlite`.

---

## 2. Syncing memory to Qdrant

There is no official MCP server for Qdrant.  A simple approach is to
use `mcp/basic-memory` and run a small Python service that periodically
reads the markdown notes and upserts them into Qdrant.

Run with:

```bash
docker-compose -f docker-compose.qdrant.yml up -d
```

### docker-compose.qdrant.yml
```yaml
version: '3.9'

volumes:
  memory-data:

networks:
  a0net: {}

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/readyz"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - a0net

  memory:
    image: mcp/basic-memory:latest
    container_name: basic-memory
    stdin_open: true
    volumes:
      - memory-data:/data
    networks:
      - a0net

  qdrant-sync:
    image: python:3.11-slim
    container_name: qdrant-sync
    command: ["python", "/sync/sync.py"]
    volumes:
      - memory-data:/memory
    depends_on:
      qdrant:
        condition: service_healthy
      memory:
        condition: service_started
    networks:
      - a0net

  agent-zero:
    image: frdel/agent-zero-run:v0.8.5
    container_name: cranky_chaplygin
    volumes:
      - ./agent-zero:/a0
    depends_on:
      - memory
    ports:
      - "50080:80"
    networks:
      - a0net
```

The `qdrant-sync` service should implement a script `sync.py` that
reads markdown from `/memory` and upserts the content into the Qdrant
instance.

---

## 3. Storing memory in Neo4j

`mcp/neo4j-memory` uses a Neo4j database as its storage backend.  Other
services can directly query Neo4j via the Bolt protocol.

Run with:

```bash
docker-compose -f docker-compose.neo4j.yml up -d
```

### docker-compose.neo4j.yml
```yaml
version: '3.9'

volumes:
  neo4j-data:
  neo4j-logs:

networks:
  a0net: {}

services:
  neo4j:
    image: neo4j:5.20
    container_name: neo4j
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j-data:/data
      - neo4j-logs:/logs
    ports:
      - "7474:7474"
      - "7687:7687"
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "password", "RETURN 1"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - a0net

  memory:
    image: mcp/neo4j-memory:latest
    container_name: neo4j-memory
    stdin_open: true
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
    depends_on:
      neo4j:
        condition: service_healthy
    networks:
      - a0net

  agent-zero:
    image: frdel/agent-zero-run:v0.8.5
    container_name: cranky_chaplygin
    volumes:
      - ./agent-zero:/a0
    depends_on:
      - memory
    ports:
      - "50080:80"
    networks:
      - a0net
```

Neo4j data and logs are persisted in volumes so other applications can
connect via Bolt or HTTP.

---

## Default compose

For reference, the default configuration in this repository
(`docker-compose.yml`) runs `mcp/memory` with its default stdio
transport and stores data in the `claude-memory` volume.  A companion
`httpd` container exposes this volume read‑only on port `4100` so other
services can fetch the memory files over HTTP.

```bash
docker-compose up -d
```

Agent Zero listens on `http://localhost:50080` and communicates with the
memory container directly via stdio. The `memory-http` service exposes
the same volume on `http://localhost:4100/` so you can download the
stored files.
