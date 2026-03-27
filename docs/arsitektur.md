## Arsitektur ProjectWise MCP Server

```mermaid
flowchart TD
    subgraph Client
        ui[UI / CLI\n(MCP Client, Inspector)]
    end

    subgraph FastAPI_Server["FastAPI Server (main.py)"]
        api[REST API Routers\n/app_kak_pipeline\n/app_product_pipeline\n/check_status_ingestion]
        mcpHttp[MCP HTTP/SSE Endpoint\n/projectwise]
    end

    subgraph MCP_Layer["MCP Server (mcp_server/server.py)"]
        fastmcp[FastMCP\n(mcp.server.fastmcp.FastMCP)]
        toolsRAG[RAG Tools\n(rag_tools.py,\nrag_pipeline.py)]
        toolsDoc[DocGen Tools\ndocgen_product_utils.py]
        toolsWeb[Web Search Tool\n(Tavily API)]
    end

    subgraph Data_Layer["Data & Storage Layer"]
        vectordb[LanceDB Vector Store\n(project & product knowledge)]
        docs[data/kak_tor,\n data/kak_tor_md,\n data/product_standard]
        templates[data/templates/proposals\nproposal_template.docx]
        generated[data/proposal_generated\n(Generated Proposals)]
    end

    subgraph External_Services["External AI Services"]
        openai[OpenAI / LLM Backend\n(OPENAI_API_KEY)]
        tavily[Tavily Web Search API\n(TAVILY_API_KEY)]
    end

    ui -->|"HTTP (REST)"| api
    ui -->|"MCP over HTTP + SSE"| mcpHttp

    api -->|"Business Logic\n(ingestion, status, dsb.)"| toolsRAG
    mcpHttp --> fastmcp

    fastmcp --> toolsRAG
    fastmcp --> toolsDoc
    fastmcp --> toolsWeb

    toolsRAG -->|"store / retrieve\nembeddings"| vectordb
    toolsRAG -->|"baca / tulis\nknowledge files"| docs

    toolsDoc --> templates
    toolsDoc --> generated

    toolsRAG -->|"LLM calls"| openai
    toolsDoc -->|"LLM calls"| openai
    toolsWeb --> tavily
```

