# RAG with Groq and LangChain

A hands-on walkthrough of building Retrieval-Augmented Generation (RAG) pipelines using **LangChain**, **Groq** (LLaMA 3.3 70B), **HuggingFace sentence-transformer embeddings**, and **FAISS** as the vector store.

This repo contains two notebooks that show the same pipeline at two stages — a baseline version and a version with retrieval tuning applied on top.

## 📓 Notebooks

### `Basic_RAG.ipynb`
Shows the core RAG idea from first principles:
1. **Without RAG** — ask the LLM (`llama-3.3-70b-versatile` via Groq) a question directly, with no external knowledge.
2. **With RAG** — embed a small set of documents into FAISS, retrieve relevant chunks for a query, and feed them to the LLM as context before it answers.
3. Compares both outputs side by side to show how grounding in retrieved context changes (and improves) the answer — including a case where the LLM only gets a fact right *because* it's present in the retrieved document.

### `Advance_RAG.ipynb`
Builds on the same pipeline with a larger document set and sets up the scaffolding for retrieval tuning — explicit `search_type` and `search_kwargs` on the retriever, ready to be extended with MMR, score thresholds, and source-document tracing.

## 🧠 Concepts covered

- **RAG fundamentals** — why retrieval grounds an LLM's answer and reduces hallucination
- **Embeddings** — using `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) to turn text into vectors
- **Vector store** — indexing documents with FAISS and querying by similarity
- **LCEL chains** — composing retriever → prompt → LLM → output parser using LangChain Expression Language (`|` pipe syntax) instead of the older `RetrievalQA` class
- **Retriever `search_type`**:
  - `"similarity"` — plain nearest-neighbor search; returns the top `k` closest chunks to your query. Simple and fast, default choice.
  - `"mmr"` (Maximal Marginal Relevance) — retrieves relevant chunks *and* filters for diversity, so you don't get `k` near-duplicate results when your knowledge base has overlapping content.
  - `"similarity_score_threshold"` — only returns chunks above a minimum similarity score, instead of always forcing exactly `k` results back (useful for saying "nothing relevant found" rather than returning weak matches).
- **`search_kwargs` tuning**:
  - `k` — how many chunks get retrieved per query. Too low → the model may miss the answer; too high → noisy/diluted context.
  - `fetch_k` — (MMR only) how many candidates to pull before the diversity filter narrows them down to `k`.
  - `lambda_mult` — (MMR only) balances relevance vs. diversity, from `0` (max diversity) to `1` (pure relevance).
- **Prompt engineering for RAG** — constraining the LLM to answer only from provided context, and to say "I don't know" when the context doesn't support an answer.
- **Groq inference** — using Groq's LPU-backed API for fast LLaMA 3.3 70B inference instead of OpenAI.

## ⚙️ Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/MuhammadSuffian/RAG_with_groq_and_langchain.git
   cd RAG_with_groq_and_langchain
   ```

2. Install dependencies:
   ```bash
   pip install langchain langchain-groq langchain-huggingface langchain-community faiss-cpu python-dotenv sentence-transformers
   ```

3. Create a `.env` file in the project root with your Groq API key:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   Get a free key at [console.groq.com](https://console.groq.com).

4. Open either notebook and run the cells top to bottom.

## 📁 Requirements

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier available)
- No OpenAI key required — embeddings run locally via HuggingFace `sentence-transformers`

## 🗺️ Roadmap

- [ ] Add `search_type="mmr"` and `"similarity_score_threshold"` examples with real output comparisons
- [ ] Return and print `source_documents` for groundedness checking
- [ ] Add real document chunking (currently uses short hardcoded strings, not a chunked corpus)
- [ ] Compare retrieval quality across different `k` / `fetch_k` / `lambda_mult` values

## 📄 License

MIT
