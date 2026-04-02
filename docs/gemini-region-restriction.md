# Gemini API: “User location is not supported”

If you see:

```text
400 FAILED_PRECONDITION … User location is not supported for the API use.
```

this comes from **Google’s policy** for the **Google AI Studio** (developer) API: requests are tied to **where they originate** (your machine’s / cloud VM’s **public IP**). Many **cloud datacenter regions** (including some in **Singapore** and elsewhere) are **not allowed** for that API, even with a valid `GEMINI_API_KEY`.

It is **not** a bug in this repo and **not** fixed by changing the model name.

---

## Recommended: local Ollama (no OpenAI / no Gemini cloud key)

Install [Ollama](https://ollama.com) on the same machine (your GPU server is fine). Pull a small English model, then run the experiment with **`--llm-backend ollama`**:

```bash
# One-time
curl -fsSL https://ollama.com/install.sh | sh   # Linux; see ollama.com for macOS/Windows
ollama serve   # if not already running as a service
ollama pull llama3.2

cd /path/to/RAG_Lab
python experiments/exp_rag_generation.py --mode all --llm-backend ollama --llm-model llama3.2
```

Optional: set `OLLAMA_BASE_URL` in `.env` if the API is not at `http://127.0.0.1:11434/v1`.

---

## Other options

1. **Run the generation experiment on a network Google allows**  
   For example: your laptop in a supported country, or a VM in a supported region.

2. **Use Vertex AI (GCP)**  
   Gemini on Vertex uses a different product surface; availability follows **GCP region** rules.

3. **`--llm-backend openai`** with `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL` for OpenRouter, Azure, etc.)

4. **SSH port-forward** — run only the LLM call from a machine where Gemini works.

Rotate any API keys if they were shared or committed by mistake.
