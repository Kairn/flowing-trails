# FlowingTrails — A VGM Composer

An AI-powered video game music generation system. Submit a creative brief and get back audio that matches the style and feel of classic VGM.

## What It Does

You describe what you want — *"intense boss fight theme, dark and orchestral, fast tempo"* — and the system generates it. Under the hood it's a multi-model agentic pipeline:

1. **Claude** parses and refines your brief into a structured music spec
2. **CLAP retrieval** searches a reference corpus for stylistically similar tracks (via Qdrant)
3. **MusicGen** generates audio conditioned on both the text spec and the retrieved audio clip as a melody reference
4. A **scoring loop** measures CLAP similarity between output and intent, and asks Claude to refine the spec if the score misses the threshold — up to 3 attempts

## Stack

| Concern               | Platform                                  |
| --------------------- | ----------------------------------------- |
| GPU inference         | Modal (A10G)                              |
| Text LLM              | Anthropic Claude API                      |
| Audio generation      | `facebook/musicgen-melody` (1.5B)         |
| Audio/text embeddings | `laion/clap-htsat-unfused`                |
| Vector database       | Qdrant Cloud                              |
| Model registry        | Hugging Face Hub                          |
| Observability         | Grafana Cloud (OTel → Tempo + Prometheus) |
| CI                    | GitHub Actions                            |
