# RAG with Groq and LangChain

A hands-on walkthrough of building Retrieval-Augmented Generation (RAG) pipelines using **LangChain**, **Groq** (LLaMA 3.3 70B), **HuggingFace sentence-transformer embeddings**, and **FAISS** as the vector store.

This repo has three notebooks that show the same pipeline across increasing levels of maturity: a baseline version, a version with retrieval tuning, and a full document Q&A agent with sourced answers.

##  Notebooks

### `Basic_RAG.ipynb`
Shows the core RAG idea from first principles:
1. **Without RAG** — ask the LLM (`llama-3.3-70b-versatile` via Groq) a question directly, with no external knowledge.
2. **With RAG** — embed a small set of documents into FAISS, retrieve relevant chunks for a query, and feed them to the LLM as context before it answers.
3. Compares both outputs side by side to show how grounding in retrieved context changes (and improves) the answer — including a case where the LLM only gets a fact right *because* it's present in the retrieved document.

### `Advance_RAG.ipynb`
Builds on the same pipeline with a larger document set and sets up the scaffolding for retrieval tuning — explicit `search_type` and `search_kwargs` on the retriever, ready to be extended with MMR, score thresholds, and source-document tracing.

### `RAG_Document_Q_A_Agent.ipynb`
Puts the tuning scaffolding to work as a full document Q&A agent over a real chunked file (`sample_faq.txt`), rather than short hardcoded strings:
1. **Real document loading and chunking** — `TextLoader` + `CharacterTextSplitter` (300 chars, 50 overlap) to split a FAQ file into retrievable chunks.
2. **MMR retrieval configured end-to-end** — `search_type="mmr"` with `k=4`, `fetch_k=20`, `lambda_mult=0.5`, so retrieved chunks are relevant but not redundant.
3. **A grounded prompt** that explicitly instructs the LLM to answer only from the provided context and say "I don't know" otherwise.
4. **An LCEL chain that returns both the answer and its sources** — built with `RunnableParallel` + `RunnablePassthrough.assign` instead of the legacy `RetrievalQA` class, so each response includes `answer` and the retrieved `context` documents for groundedness checking.
5. **Batch Q&A over multiple questions** — loops through a list of FAQ-style questions, printing the answer and supporting source chunks for each.

##  Concepts covered

- **RAG fundamentals** — why retrieval grounds an LLM's answer and reduces hallucination
- **Embeddings** — using `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) to turn text into vectors
- **Vector store** — indexing documents with FAISS and querying by similarity
- **Document chunking** — loading real text files and splitting them into overlapping chunks with `CharacterTextSplitter`
- **LCEL chains** — composing retriever → prompt → LLM → output parser using LangChain Expression Language (`|` pipe syntax) instead of the older `RetrievalQA` class
- **Returning sources alongside answers** — using `RunnableParallel` and `RunnablePassthrough.assign` to carry retrieved source documents through the chain for groundedness checking
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

##  Setup

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

4. Open any notebook and run the cells top to bottom.

##  Requirements

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier available)
- No OpenAI key required — embeddings run locally via HuggingFace `sentence-transformers`

##  Roadmap

- [x] Add real document chunking (now implemented in `RAG_Document_Q_A_Agent.ipynb` via `sample_faq.txt`)
- [x] Add `search_type="mmr"` example with real output comparisons (now implemented in `RAG_Document_Q_A_Agent.ipynb`)
- [x] Return and print `source_documents` for groundedness checking (now implemented in `RAG_Document_Q_A_Agent.ipynb`)
- [ ] Add `"similarity_score_threshold"` example with real output comparisons
- [ ] Compare retrieval quality across different `k` / `fetch_k` / `lambda_mult` values

##  License

MIT
