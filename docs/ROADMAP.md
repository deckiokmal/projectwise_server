## Backend
- Ujicoba vLLM Inference dan Integrasikan ke Flask
- Ujicoba gpt-oss:20b (MoE model) menggunakan vLLM Inference
- Research bagaimana cara mengintegrasikan vLLM sebagai OpenAI-compatible API
    - Tool Choice: vLLM hanya mendukung named function calling. Opsi tool_choice="auto" atau "required" belum tersedia.
    - Arsitektur best practice vLLM production:
    ```mermaid
    graph LR
        Client -->|HTTP/gRPC| Flask_API
        Flask_API -->|HTTP POST| vLLM_Service
        subgraph Infrastructure
            vLLM_Service --> GPU_Node1
            vLLM_Service --> GPU_Node2
            Flask_API --> CPU_Node
        end
    ```
- Menggunakan entrypoint.sh saat build docker image
