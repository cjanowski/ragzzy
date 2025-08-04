# RagZzy — Retrieval‑Augmented Support Assistant

RagZzy is a modern, retrieval‑augmented support assistant that blends semantic + keyword retrieval, dynamic few‑shot prompting, and a post‑generation validator with an optional single‑revision loop. It’s built to demonstrate strong product thinking, pragmatic engineering, and measurable quality.

VIST HERE: [RagZzy](https://ragzzy.vercel.app/)

## Why This Is Interesting

- Context-aware RAG: hybrid retrieval combining semantic embeddings and BM25‑ish keyword search with contextual and entity boosts.
- Dynamic few-shot: automatically selects the most relevant support exemplars from curated seeds to steer tone and structure.
- Validator with revision: enforces response quality using a checklist and performs a single self‑revision when needed.
- Offline eval CLI: run local evaluations against seeds, report structure adherence and coverage metrics, export JSONL for analysis.

This project showcases: retrieval architectures, prompt engineering, safety/consistency checks, async streaming, and pragmatic tooling for evaluation.
<img width="1512" height="822" alt="Screenshot 2025-08-04 at 6 49 33 PM" src="https://github.com/user-attachments/assets/967041d5-6710-40bb-951f-eb1462e0b80a" />

---

## Features

1) Hybrid Retrieval with Sentence Packing
- Semantic retrieval (Gemini embeddings) + MiniSearch keyword search.
- Contextual and entity alignment boosts from prior conversation.
- Sentence-level packing under a token budget for high signal density.

2) Senior Support Persona
- Persona block with principles, tone, and structured format guidance.
- Optional enablement via SUPPORT_PERSONA or per-request override.
- Consistency without over-constraining domain-specific content.

3) Dynamic Few‑Shot Selection
- Seeds defined in scripts/seedSupportPersona.js (id, user, assistant, tags, difficulty).
- Embeds user query and seeds, ranks by cosine similarity, selects top‑K.
- Injects only the best examples into the prompt to stay within budget.

4) Post‑Generation Validator + Single Revision
- Checklist-driven validation (sections: Summary, Steps, Validation, Rollback, Notes).
- Strict JSON verdict parse with fallback for robustness.
- If violations found, performs one revision pass to satisfy the checklist.

5) Offline Eval CLI
- Run seeds through the pipeline, compute metrics, and export JSONL.
- Metrics include structure adherence, length tokens, tag coverage, and retrieval stats.
- Configurable flags for K, similarity threshold, persona enablement, and output.

---

## Architecture Overview

- API entry: [api.chat.module.exports()](api/chat.js:143) handles POST with optional streaming.
- Query analysis: intent classification, entity extraction, follow‑up detection, light query expansion.
- Retrieval: [api.chat.performContextAwareRetrieval()](api/chat.js:1188) runs semantic + keyword + contextual search, then packs sentences under a token budget.
- Prompt building: [api.chat.buildDynamicPrompt()](api/chat.js:1489) assembles persona, dynamic few‑shots, conversation history, retrieved context, and instructions.
- Generation: Gemini 2.0 Flash (configurable), streaming or non‑streaming.
- Validation: [api.chat.runValidatorAndMaybeRevise()](api/chat.js:2219) enforces checklist and revises once if needed.
- Caching: embedding caches for knowledge chunks and few‑shot seeds.

Key files:
- [api.chat.generateContextAwareResponse()](api/chat.js:1429): non‑streaming generation + validator pass.
- [api.chat.buildDynamicPrompt()](api/chat.js:1489): persona + dynamic few‑shot + context assembly.
- [api.persona.getSupportValidatorChecklist()](api/persona.js:1): checklist provider.
- [scripts/seedSupportPersona.js](scripts/seedSupportPersona.js:1): curated support examples used for dynamic few‑shot.
- [scripts/eval.js](scripts/eval.js:1): offline evaluator.

---

## Setup

Requirements:
- Node.js 18+ recommended.
- A Google AI Studio API key (GEMINI_API_KEY).

Install:
- npm install

Environment:
- Export your Gemini API key:
  - macOS/Linux: export GEMINI_API_KEY=YOUR_KEY
  - Windows (PowerShell): setx GEMINI_API_KEY "YOUR_KEY" (restart terminal)
- Optional: enable persona globally
  - export SUPPORT_PERSONA=1

---

## Running Locally

Non‑streaming POST endpoint is at [api.chat.module.exports()](api/chat.js:143). This project targets serverless adapters; for local usage, wire your server or simulate via scripts.

Frontend example client is in [public/script.js](public/script.js:1) (streams SSE tokens).

---

## Dynamic Few‑Shot Configuration

In [api/chat.js](api/chat.js:99):
- fewshot.enable: boolean (default true)
- fewshot.k: number of examples (env FS_K supported)
- fewshot.maxTokens: budget for few‑shot block
- fewshot.minSimilarity: filter threshold

Seeds live in [scripts/seedSupportPersona.js](scripts/seedSupportPersona.js:1). The selector:
- Embeds the user query and seed concatenation (user + assistant).
- Scores by cosine similarity and picks top‑K (fallback to top‑K if all below threshold).
- Trims under token budget.

---

## Validator and Single‑Revision

Checklist source:
- [api.persona.getSupportValidatorChecklist()](api/persona.js:1)

Flow:
- Model returns JSON verdict { ok, missing, notes }.
- If ok=false, a single revision pass runs with persona + checklist to fix omissions while preserving correct content.
- Validation meta is attached to the response.

---

## Offline Evaluation

CLI: [scripts/eval.js](scripts/eval.js:1)

Prerequisites:
- export GEMINI_API_KEY=YOUR_KEY

Quick start:
- npm run eval
- node scripts/eval.js --limit 10
- With persona on: SUPPORT_PERSONA=1 node scripts/eval.js --persona --limit 10

Useful flags:
- --limit N: number of seeds
- --k K: few‑shot top‑K (or set FS_K)
- --minSim 0.2: minimum similarity threshold
- --persona: force-enable senior support persona for the run
- --no-validate: bypass validator pass
- --out eval.jsonl: write per‑seed JSONL

NPM scripts (see [package.json](package.json:6)):
- npm run eval
- npm run eval:persona
- npm run eval:k5
- npm run eval:out

Outputs:
- Per‑seed JSON (JSONL when --out used) containing:
  - retrieval stats: bestSimilarity, averageScore, chunk scores
  - analysis: intent, entities
  - response: text, lengthTokens, structureOk, missing sections, tag coverage
  - validation: revised or not, verdict payload
- Summary:
  - total, structureOkRatePct, avgResponseTokens, avgRetrievalBestSimilarity, avgTagCoverage

---

## Deployment Notes

- Set GEMINI_API_KEY in your hosting environment.
- Optionally set SUPPORT_PERSONA=1 to enable persona globally; clients can override per‑request.
- The code handles streaming (SSE) and non‑streaming responses. Ensure your platform passes through SSE headers.

---

## Design Decisions and Trade‑offs

- Gemini for both embeddings and generation: fewer dependencies, consistent embeddings, single‑vendor simplicity.
- Hybrid retrieval: semantic (broad recall) plus MiniSearch (exact/rare terms).
- Sentence packing: improves information density under prompt budgets.
- Single revision only: avoids loops; predictable latency and cost.
- Dynamic few‑shot only: removed static few‑shot duplication to save tokens and reduce drift.

---

## What Employers Should Notice

- Clear separation of concerns and pragmatic guards around async, streaming, and validation.
- Measurable quality via the offline evaluator with structured metrics.
- Thoughtful defaults and configuration knobs for persona, few‑shot, and retrieval budgets.
- Safe fallbacks (JSON parse guards, empty KB handling, MiniSearch fallbacks).

---

## Roadmap Ideas

- Add golden answer assertions to eval (precision/recall with pattern banks).
- Broaden safety checks with a full moderation pass and red‑team seeds.
- Export more internals for reuse (e.g., prompt builder) to avoid duplication in CLI.
- Persist seed/embedding caches across runs (on-disk).
- Expand personas and per‑intent response styles.

---

## License

ISC
