#!/usr/bin/env node

/**
 * Offline Eval CLI for RagZzy Support Persona
 *
 * Runs seed prompts through the same pipeline used by the API:
 *  - analyzeQuery
 *  - performContextAwareRetrieval
 *  - dynamic prompt (with persona + dynamic few-shot)
 *  - model generation
 *  - validator + single revision pass
 *
 * Outputs per-seed JSONL and prints a summary to stdout.
 *
 * Environment:
 *  - GEMINI_API_KEY must be set
 *  - SUPPORT_PERSONA=true to enable persona behaviors (or --persona flag)
 *
 * Usage:
 *  node scripts/eval.js [--limit N] [--k K] [--no-validate] [--out output.jsonl] [--persona] [--minSim 0.2]
 */

const fs = require('fs');
const path = require('path');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// Pull selected internals from the chat API to maximize reuse
const chatApi = require(path.join(process.cwd(), 'api', 'chat.js'));

// Try to load seeds
let seeds = [];
try {
  seeds = require(path.join(process.cwd(), 'scripts', 'seedSupportPersona.js'));
} catch (e) {
  console.warn('Could not load scripts/seedSupportPersona.js:', e?.message);
  process.exitCode = 1;
  console.error('No seeds found to evaluate.');
  process.exit(1);
}

// Basic arg parsing
const args = process.argv.slice(2);
function getFlag(name, fallback = undefined) {
  const idx = args.findIndex(a => a === `--${name}`);
  if (idx !== -1) {
    const next = args[idx + 1];
    if (!next || next.startsWith('--')) return true; // boolean flag
    return next;
  }
  return fallback;
}
const limit = parseInt(getFlag('limit', '0'), 10);
const k = parseInt(getFlag('k', process.env.FS_K || '3'), 10);
const noValidate = !!getFlag('no-validate', false);
const outPath = getFlag('out', '');
const personaFlag = !!getFlag('persona', false);
const minSim = parseFloat(getFlag('minSim', '0.2'));

// Ensure API key
if (!process.env.GEMINI_API_KEY) {
  console.error('GEMINI_API_KEY is not set. Please export it before running.');
  process.exit(1);
}

// Prepare output stream if requested
let outStream = null;
if (outPath) {
  const absOut = path.isAbsolute(outPath) ? outPath : path.join(process.cwd(), outPath);
  outStream = fs.createWriteStream(absOut, { flags: 'w' });
  console.log(`Writing JSONL to ${absOut}`);
}

// Construct a minimal conversation context for each run
function makeContext() {
  return {
    sessionId: `eval_${Math.random().toString(36).slice(2)}`,
    messages: [],
    entities: new Map(),
    topics: [],
    userPreferences: {},
    createdAt: Date.now(),
    lastActivity: Date.now(),
    intentHistory: [],
    successfulResponses: 0,
    totalResponses: 0,
    _requestPersona: personaFlag ? 'senior_support' : undefined
  };
}

function estimateTokens(text) {
  // Heuristic matches api/chat.js usage
  const perChar = 0.25;
  return Math.ceil((text?.length || 0) * perChar);
}

// Simple structural adherence checks for the support format
function structuralChecks(text) {
  const mustHave = ['Summary', 'Steps', 'Validation', 'Rollback', 'Notes'];
  const missing = mustHave.filter(h => !new RegExp(`\\b${h}\\b`, 'i').test(text));
  return {
    ok: missing.length === 0,
    missing
  };
}

async function runOneSeed(seed, index) {
  const requestId = `eval_req_${Date.now()}_${index}`;
  const conversationContext = makeContext();

  // Ensure knowledge base is ready
  await chatApi.ensureKnowledgeBase();

  // Analyze
  const analysis = await chatApi.analyzeQuery(seed.user, conversationContext, requestId);

  // Retrieval
  const retrieval = await chatApi.performContextAwareRetrieval(analysis, conversationContext, requestId);

  // Build prompt using the same dynamic builder in api/chat.js by invoking the same path used in API
  // We call the same model as chat API to get a response. To maximize reuse, we reimplement the minimal call here
  // by requiring api/chat.js's internal run via generateContent with the exact dynamic prompt.
  // Since buildDynamicPrompt is internal, we reconstruct it using the same path as generateContextAwareResponse:
  // For strict reuse, we invoke the public function generateEmbedding etc., but we still need a prompt.
  // We'll derive the prompt by instantiating a small inlined builder that mirrors api/chat.js behavior,
  // or we can call the comprehensive candidate path indirectly by triggering the model call with similar params.
  //
  // Simpler path: Reuse the chat endpoint "generateContextAwareResponse" by simulating candidate generation step.
  // However, that function is not exported. We'll inline a minimal mirror of the prompt construction
  // by leveraging the same helper behavior: history, context, and persona is handled by the API call.
  //
  // Instead of duplicating logic, we'll construct the same inputs and call the model ourselves with a concise prompt.

  // Pull in persona tools from api/persona.js for rendering few-shot and persona block
  const {
    buildSeniorSupportPersonaBlock,
    renderFewShotExamples,
    getSupportValidatorChecklist
  } = require(path.join(process.cwd(), 'api', 'persona.js'));

  // For few-shot selection, reuse selector from api/chat.js via module import side-effect (function is not exported).
  // We do a lightweight reimplementation here consistent with api/chat.js behavior by calling its exported generateEmbedding.
  async function embedWithCache(text) {
    return await chatApi.generateEmbedding(text);
  }
  async function cosine(a, b) {
    // simple cosine from api/chat.js logic replicated here
    if (!a || !b || a.length !== b.length) return 0;
    let dot = 0, ma = 0, mb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      ma += a[i] * a[i];
      mb += b[i] * b[i];
    }
    const denom = Math.sqrt(ma) * Math.sqrt(mb);
    return denom === 0 ? 0 : dot / denom;
  }
  const seedEmbCache = new Map();
  async function getSeedEmb(seed) {
    const key = seed.id || seed.user.slice(0, 64);
    if (seedEmbCache.has(key)) return seedEmbCache.get(key);
    const text = `${seed.user}\n${seed.assistant}`.slice(0, 2000);
    const emb = await embedWithCache(text);
    seedEmbCache.set(key, emb);
    return emb;
  }
  async function selectFewShotSeeds(query, seedsArr, topK = 3, minSimLocal = 0.2) {
    const qEmb = await embedWithCache(query);
    const scored = [];
    for (const s of seedsArr) {
      try {
        const e = await getSeedEmb(s);
        const sim = await cosine(qEmb, e);
        scored.push({ seed: s, sim });
      } catch {}
    }
    scored.sort((a, b) => b.sim - a.sim);
    const picked = scored.filter(x => x.sim >= minSimLocal).slice(0, topK).map(x => x.seed);
    return picked.length > 0 ? picked : scored.slice(0, topK).map(x => x.seed);
  }

  const contextBlock = retrieval.chunks.map(c => `- ${c.chunk.text}`).join('\n');
  const history = conversationContext.messages
    .slice(-6)
    .map(msg => `${msg.type === 'user' ? 'User' : 'Assistant'}: ${msg.content}`)
    .join('\n');

  const enablePersona = personaFlag || (process.env.SUPPORT_PERSONA === 'true' || process.env.SUPPORT_PERSONA === '1');

  const sections = [];
  if (enablePersona) {
    sections.push(buildSeniorSupportPersonaBlock());
    // dynamic few-shot
    const selected = await selectFewShotSeeds(analysis.originalQuery, seeds, isFinite(k) && k > 0 ? k : 3, isFinite(minSim) ? minSim : 0.2);
    let fewShotStr = renderFewShotExamples(selected.map(s => ({ user: s.user, assistant: s.assistant }))) || '';
    // enforce budget ~ matches api heuristic
    const maxFewshotTokens = 600;
    if (estimateTokens(fewShotStr) > maxFewshotTokens) {
      // crude truncation at paragraph boundaries
      const parts = fewShotStr.split('\n\n');
      let acc = [];
      for (const p of parts) {
        const tentative = acc.concat([p]).join('\n\n');
        if (estimateTokens(tentative) > maxFewshotTokens) break;
        acc.push(p);
      }
      fewShotStr = acc.join('\n\n');
    }
    if (fewShotStr) sections.push(fewShotStr);
  }

  sections.push('You are RagZzy, a helpful customer support assistant.');
  if (history) sections.push(`Previous conversation:\n${history}`);
  if (contextBlock) sections.push(`Relevant information:\n${contextBlock}`);

  const parts = [`Current question: ${analysis.originalQuery}`];
  if (analysis.contextualReferences.length > 0) {
    parts.push(
      `Note: The user used references like "${analysis.contextualReferences.map(ref => ref.pronoun).join(', ')}" which may refer to previous discussion topics.`
    );
  }
  sections.push(parts.join('\n'));
  sections.push('Provide a detailed, thorough response that fully addresses the question. Include relevant details and context.');
  sections.push('Response:');

  const prompt = sections.join('\n\n');

  // Model call
  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  const chatModel = genAI.getGenerativeModel({ model: 'gemini-2.0-flash-exp' });
  const generation = await chatModel.generateContent({
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig: {
      maxOutputTokens: 800,
      temperature: 0.4,
      topP: 0.8,
      topK: 40
    }
  });
  const rawText = generation.response.text() || '';

  // Validator
  let finalText = rawText;
  let validation = {
    validated: false,
    revised: false,
    verdict: null,
    error: null
  };

  if (!noValidate) {
    // Reuse api/chat.js validator by feeding through the exported pipeline where possible.
    // runValidatorAndMaybeRevise is internal, so replicate a minimal call here using persona's checklist.
    const checklist = enablePersona
      ? require(path.join(process.cwd(), 'api', 'persona.js')).getSupportValidatorChecklist()
      : ['Checklist:', '- Answer is helpful and accurate.', '- No unsafe content.', '- Responds directly to the question.'].join('\n');

    const judgePrompt = [
      'You are a strict validator. Given the Assistant response below, check the checklist and return a strict JSON object:',
      '{ "ok": boolean, "missing": string[], "notes": string }',
      '',
      checklist,
      '',
      'Assistant response:',
      rawText
    ].join('\n');

    try {
      const evalRes = await chatModel.generateContent({
        contents: [{ role: 'user', parts: [{ text: judgePrompt }] }],
        generationConfig: { maxOutputTokens: 300, temperature: 0.1, topP: 0.8, topK: 40 }
      });
      const evalText = evalRes.response.text() || '';
      let verdict = { ok: true, missing: [], notes: '' };
      try {
        const jsonMatch = evalText.match(/\{[\s\S]*\}/);
        if (jsonMatch) verdict = JSON.parse(jsonMatch[0]);
      } catch {
        verdict = { ok: true, missing: [], notes: 'parse_failed' };
      }

      if (!verdict.ok) {
        const revisePrompt = [
          enablePersona ? buildSeniorSupportPersonaBlock() : 'You are a helpful assistant.',
          '',
          'Revise the response to satisfy ALL checklist items below while preserving correct content.',
          checklist,
          '',
          'Original response:',
          rawText,
          '',
          'Revised response:'
        ].join('\n');

        const revRes = await chatModel.generateContent({
          contents: [{ role: 'user', parts: [{ text: revisePrompt }] }],
          generationConfig: { maxOutputTokens: 900, temperature: 0.3, topP: 0.8, topK: 40 }
        });
        finalText = revRes.response.text() || rawText;
        validation = { validated: true, revised: true, verdict, error: null };
      } else {
        validation = { validated: true, revised: false, verdict, error: null };
      }
    } catch (e) {
      validation = { validated: false, revised: false, verdict: null, error: 'validator_failed' };
    }
  }

  // Metrics
  const structure = structuralChecks(finalText);
  const lengthTokens = estimateTokens(finalText);
  const tagCoverage = Array.isArray(seed.tags) && seed.tags.length
    ? seed.tags.filter(t => new RegExp(`\\b${escapeRegex(t)}\\b`, 'i').test(finalText)).length / seed.tags.length
    : null;

  const record = {
    id: seed.id || `seed_${index}`,
    idx: index,
    user: seed.user,
    expectedStyle: 'support_persona',
    tags: seed.tags || [],
    difficulty: seed.difficulty || null,
    retrieval: {
      chunks: retrieval.chunks.map(c => ({ id: c.chunk.id, source: c.metadata?.source || 'knowledge_base', score: c.score })),
      bestSimilarity: retrieval.bestSimilarity,
      contextBoost: retrieval.contextBoost,
      averageScore: retrieval.averageScore
    },
    analysis: {
      intent: analysis.intent,
      intentConfidence: analysis.intentConfidence,
      entities: analysis.entities
    },
    response: {
      text: finalText,
      lengthTokens,
      structureOk: structure.ok,
      structureMissing: structure.missing,
      tagCoverage
    },
    validation
  };

  if (outStream) outStream.write(JSON.stringify(record) + '\n');

  return record;
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

async function main() {
  const total = (limit && limit > 0) ? Math.min(seeds.length, limit) : seeds.length;
  console.log(`Evaluating ${total} seeds (persona=${personaFlag}, k=${k}, minSim=${minSim}, validate=${!noValidate})`);
  const results = [];
  let okCount = 0;

  for (let i = 0; i < total; i++) {
    const r = await runOneSeed(seeds[i], i);
    results.push(r);
    if (r.response.structureOk) okCount++;

    // Simple progress log
    console.log(`[${i + 1}/${total}] ${r.id} structureOk=${r.response.structureOk} tagCoverage=${r.response.tagCoverage ?? 'n/a'} tokens=${r.response.lengthTokens}`);
  }

  // Summary
  const structureRate = (okCount / total) * 100;
  const avgTokens = (results.reduce((s, r) => s + (r.response.lengthTokens || 0), 0) / Math.max(1, results.length)).toFixed(1);
  const avgBestSim = (results.reduce((s, r) => s + (r.retrieval.bestSimilarity || 0), 0) / Math.max(1, results.length)).toFixed(3);
  const avgTagCoverage = (results
    .map(r => (r.response.tagCoverage == null ? null : r.response.tagCoverage))
    .filter(v => v != null)
    .reduce((s, v, _, arr) => s + v / arr.length, 0)).toFixed(3);

  const summary = {
    total,
    structureOkRatePct: Number(structureRate.toFixed(1)),
    avgResponseTokens: Number(avgTokens),
    avgRetrievalBestSimilarity: Number(avgBestSim),
    avgTagCoverage: isNaN(Number(avgTagCoverage)) ? null : Number(avgTagCoverage),
    timestamp: Date.now()
  };

  console.log('\nSummary:');
  console.log(JSON.stringify(summary, null, 2));

  if (outStream) outStream.end();
}

main().catch(err => {
  console.error('Eval failed:', err);
  process.exit(1);
});