/**
 * RagZzy Chat API - Main chat endpoint with RAG functionality
 * Handles user queries, retrieval-augmented generation, and knowledge contribution prompts
 */

const { GoogleGenerativeAI } = require('@google/generative-ai');
const fs = require('fs');
const path = require('path');
const MiniSearch = require('minisearch');
const {
    truthyEnv,
    buildSeniorSupportPersonaBlock,
    getFewShotExamples,
    renderFewShotExamples,
    getSupportValidatorChecklist
} = require('./persona');

// Initialize Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const chatModel = genAI.getGenerativeModel({ model: 'gemini-2.0-flash-exp' });
const embeddingModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });

// Load support seeds for dynamic few-shot selection
let supportSeeds = [];
try {
    supportSeeds = require(path.join(process.cwd(), 'scripts', 'seedSupportPersona.js'));
} catch (e) {
    console.warn('Could not load scripts/seedSupportPersona.js for few-shot selection:', e?.message);
}

// Streaming support: helper to convert model stream chunks to text parts
async function streamModelToResponse(model, prompt, res, requestId, generationConfig = {}) {
    // Server-Sent Events headers if not already set by platform
    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');

    // Small helper to send SSE data frames
    const send = (event, data) => {
        try {
            res.write(`event: ${event}\n`);
            res.write(`data: ${JSON.stringify(data)}\n\n`);
        } catch (e) {
            console.error(`[${requestId}] SSE write error:`, e?.message);
        }
    };

    let full = '';
    let emittedAnyToken = false;

    try {
        // gemini streaming API
        const start = Date.now();
        const stream = await model.generateContentStream({
            contents: [{ role: 'user', parts: [{ text: prompt }] }],
            generationConfig: {
                maxOutputTokens: generationConfig.maxOutputTokens ?? 800,
                temperature: generationConfig.temperature ?? 0.5,
                topP: generationConfig.topP ?? 0.8,
                topK: generationConfig.topK ?? 40
            }
        });

        for await (const chunk of stream.stream) {
            const text = chunk?.text() || '';
            if (text) {
                full += text;
                send('token', { token: text });
                emittedAnyToken = true;
            }
        }

        performanceMonitor.recordGeneration(Date.now() - start);
        send('done', { complete: true });

        return full;
    } catch (err) {
        console.error(`[${requestId}] Streaming error:`, err);
        // If nothing was emitted, emit a minimal friendly token to avoid blank bubble
        if (!emittedAnyToken) {
            const fallback = "I'm having trouble generating a response right now. Please try again in a moment.";
            full = fallback;
            try {
                send('token', { token: fallback });
                send('done', { complete: true });
            } catch (_e) {}
            // Return gracefully without throwing so caller flow can continue
            return full;
        }
        // If some tokens already emitted, try to close cleanly
        try { send('done', { complete: true }); } catch (_e) {}
        // Return what we have
        return full;
    }
}

// In-memory knowledge base cache
let knowledgeBase = null;
let lastProcessed = 0;

// MiniSearch index (BM25-ish keyword retrieval)
let miniSearch = null;
let miniSearchReady = false;

// In-memory conversation memory and user personalization
const conversationMemory = new Map(); // sessionId -> conversation data
const userProfiles = new Map(); // userId -> user preferences and learning data
const responseQualityTracker = new Map(); // responseId -> quality metrics

// Conversation context storage (in-memory - would use Redis/database in production)
const conversationContexts = new Map();
const SESSION_TIMEOUT = 30 * 60 * 1000; // 30 minutes
const MAX_CONTEXT_MESSAGES = 10; // Keep last 10 messages per session

// Configuration
const CONFIG = {
    confidenceThreshold: parseFloat(process.env.CONFIDENCE_THRESHOLD) || 0.3,
    maxRetrievalChunks: 5,
    chunkSize: 500,
    chunkOverlap: 50,
    maxResponseTokens: 1000,
    // New context-aware configurations
    maxCandidateResponses: 3,
    contextWeight: 0.3,
    intentConfidenceThreshold: 0.6,
    queryExpansionEnabled: true,
    adaptiveChunkRetrieval: true,
    maxQueryExpansions: 2,
    // Feature flags
    supportPersonaEnabled: truthyEnv(process.env.SUPPORT_PERSONA),
    // Hybrid retrieval weights and flags
    hybrid: {
        enableMiniSearch: true,
        semanticWeight: 0.6,
        keywordWeight: 0.35,
        contextualWeight: 0.1,
        multiTypeBoost: 1.2,
        freshnessNovalue: 1.0 // fallback boost if not present
    },
    // Dynamic few-shot using seeds
    fewshot: {
        enable: true,
        k: parseInt(process.env.FS_K || '3', 10),
        maxTokens: 600, // budget for examples block
        minSimilarity: 0.2
    },
    // Sentence-level packing configuration
    packing: {
        enableSentencePacking: true,
        maxContextTokens: 1500, // budget for retrieved context before prompt
        sentenceSplitRegex: /(?<=\\.|\\?|!)\\s+(?=[A-Z0-9])/,
        tokensPerChar: 0.25 // rough heuristic for Gemini tokens
    }
};

/**
 * Main request handler
 */
module.exports = async (req, res) => {
    // Set CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
    if (req.method === 'OPTIONS') {
        return res.status(200).end();
    }

    // Support GET for simple health of this route and streaming preflight tests
    if (req.method === 'GET') {
        return res.status(200).json({
            ok: true,
            route: 'chat',
            streaming: true,
            persona: {
                supportPersonaEnabled: CONFIG.supportPersonaEnabled,
                overrideParam: 'persona', // send "senior_support" to enable per-request
                acceptedValues: ['senior_support', null]
            }
        });
    }
    
    if (req.method !== 'POST') {
        return res.status(405).json({
            error: 'Method not allowed',
            code: 'METHOD_NOT_ALLOWED'
        });
    }
    
    const startTime = Date.now();
    const requestId = generateRequestId();
    
    try {
        // Validate and parse request
        const { message, sessionId, stream, persona } = req.body;

        // Early guard: if client requested stream but the model key is missing, return JSON error
        if (stream && !process.env.GEMINI_API_KEY) {
            return res.status(503).json({
                error: 'AI service not configured',
                code: 'AI_CONFIG_MISSING',
                requestId,
                retryable: false
            });
        }
        
        if (!message || typeof message !== 'string') {
            return res.status(400).json({
                error: 'Message is required',
                code: 'INVALID_INPUT',
                requestId
            });
        }
        
        if (message.length > 1000) {
            return res.status(400).json({
                error: 'Message too long (max 1000 characters)',
                code: 'MESSAGE_TOO_LONG',
                requestId
            });
        }
        
        if (message.trim().length < 1) {
            return res.status(400).json({
                error: 'Message cannot be empty',
                code: 'EMPTY_MESSAGE',
                requestId
            });
        }

        // Basic input safety/sanitization filter
        const safety = basicSafetyGuard(message);
        if (!safety.allowed) {
            return res.status(400).json({
                error: 'Message rejected by safety rules',
                code: 'SAFETY_BLOCKED',
                reason: safety.reason,
                requestId
            });
        }
        
        // Initialize or update knowledge base
        await ensureKnowledgeBase();
        
        // Process chat request (non-streaming path first)
        if (!stream) {
            const response = await processChatRequest(message, sessionId, requestId);
            const processingTime = Date.now() - startTime;
            
            performanceMonitor.recordRequest(true, processingTime);
            
            return res.status(200).json({
                ...response,
                timestamp: Date.now(),
                processingTime,
                requestId,
                systemMetrics: {
                    embeddingCacheHits: embeddingCache.size,
                    activeSessions: conversationContexts.size,
                    averageResponseTime: performanceMonitor.metrics.requests.averageResponseTime
                }
            });
        }

        // Streaming path: reuse the internal pipeline to build prompt/context,
        // but emit tokens as they are generated.
        const conversationContext = getOrCreateConversationContext(sessionId, requestId);
        const queryAnalysis = await analyzeQuery(message, conversationContext, requestId);
        const retrievalResults = await performContextAwareRetrieval(queryAnalysis, conversationContext, requestId);
        // Optional per-request persona override (does not persist)
        if (typeof persona === 'string') {
            conversationContext._requestPersona = persona;
        }

        // Build a deterministic dynamic prompt with slightly lower temperature for streaming
        const contextBlock = retrievalResults.chunks.map(c => `- ${c.chunk.text}`).join('\n');
        const history = conversationContext.messages
            .slice(-6)
            .map(msg => `${msg.type === 'user' ? 'User' : 'Assistant'}: ${msg.content}`)
            .join('\n');
        const prompt = await buildDynamicPrompt(queryAnalysis, contextBlock, history, 'comprehensive', conversationContext);

        const streamedText = await streamModelToResponse(chatModel, prompt, res, requestId, {
            maxOutputTokens: getMaxTokensForResponseType('comprehensive'),
            temperature: 0.4,
            topP: 0.8,
            topK: 40
        });

        // After stream completes, compute confidence and push to context once.
        const sanitizedStreamText = sanitizeResponseText(streamedText);
        const finalConfidence = calculateResponseConfidence(retrievalResults, queryAnalysis, sanitizedStreamText);
        updateConversationContext(conversationContext, message, {
            text: sanitizedStreamText,
            confidence: finalConfidence,
            sources: retrievalResults.chunks.map(chunk => chunk.metadata?.source || 'knowledge_base').filter(Boolean)
        }, queryAnalysis);

        // End the response explicitly in some serverless runtimes
        try { res.end(); } catch {}

    } catch (error) {
        console.error(`[${requestId}] Chat API error:`, error);
        
        const processingTime = Date.now() - startTime;
        
        const errorResponse = categorizeError(error, requestId, processingTime);
        
        performanceMonitor.recordRequest(false, processingTime);
        performanceMonitor.recordError(errorResponse.body.code);
        
        // If we were in SSE mode, try to emit as SSE error
        try {
            res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
            res.write(`event: error\n`);
            res.write(`data: ${JSON.stringify(errorResponse.body)}\n\n`);
            try { res.end(); } catch {}
            return;
        } catch (_e) {
            return res.status(errorResponse.status).json(errorResponse.body);
        }
    }
};

/**
 * Process the chat request using enhanced context-aware AI
 */
async function processChatRequest(message, sessionId, requestId) {
    try {
        // Step 1: Get or create conversation context
        const conversationContext = getOrCreateConversationContext(sessionId, requestId);
        
        // Step 2: Advanced query processing
        const queryAnalysis = await analyzeQuery(message, conversationContext, requestId);
        
        // Step 3: Context-aware retrieval with query expansion
        const retrievalResults = await performContextAwareRetrieval(queryAnalysis, conversationContext, requestId);
        
        // Step 4: Generate multiple candidate responses
        const candidateResponses = await generateCandidateResponses(
            queryAnalysis, 
            retrievalResults, 
            conversationContext, 
            requestId
        );
        
        // Step 5: Select and rank best response
        const finalResponse = await selectBestResponse(candidateResponses, queryAnalysis, conversationContext);
        
        // Step 6: Update conversation context
        updateConversationContext(conversationContext, message, finalResponse, queryAnalysis);
        
        // Step 7: Determine contribution prompt need
        const shouldPromptContribution = finalResponse.confidence < CONFIG.confidenceThreshold && 
                                       message.length > 10 && 
                                       !isSpamOrInappropriate(message);
        
        const result = {
            response: finalResponse.text,
            confidence: finalResponse.confidence,
            sources: finalResponse.sources,
            intent: queryAnalysis.intent,
            processingMetadata: {
                queryAnalysis: {
                    intent: queryAnalysis.intent,
                    intentConfidence: queryAnalysis.intentConfidence,
                    entities: queryAnalysis.entities,
                    expandedQueries: queryAnalysis.expandedQueries
                },
                retrieval: {
                    chunksRetrieved: retrievalResults.chunks.length,
                    bestSimilarity: retrievalResults.bestSimilarity,
                    contextBoost: retrievalResults.contextBoost
                },
                response: {
                    candidatesGenerated: candidateResponses.length,
                    selectionReason: finalResponse.selectionReason,
                    qualityScore: finalResponse.qualityScore
                }
            }
        };
        
        // Add contribution prompt if needed
        if (shouldPromptContribution) {
            result.contributionPrompt = {
                show: true,
                message: "I don't have enough information to fully answer that question. Would you like to help by providing the answer?",
                suggestedPrompts: getRelevantGuidedPrompts(message)
            };
        } else {
            result.contributionPrompt = {
                show: false
            };
        }
        
        return result;
        
    } catch (error) {
        console.error(`[${requestId}] Enhanced RAG processing error:`, error);
        throw error;
    }
}

/**
 * Get or create conversation context for a session
 */
function getOrCreateConversationContext(sessionId, requestId) {
    const now = Date.now();
    
    // Clean up expired sessions
    cleanupExpiredSessions(now);
    
    if (!sessionId) {
        sessionId = generateSessionId();
    }
    
    if (!conversationContexts.has(sessionId)) {
        conversationContexts.set(sessionId, {
            sessionId,
            messages: [],
            entities: new Map(),
            topics: [],
            userPreferences: {},
            createdAt: now,
            lastActivity: now,
            intentHistory: [],
            successfulResponses: 0,
            totalResponses: 0
        });
        console.log(`[${requestId}] Created new conversation context for session: ${sessionId}`);
    }
    
    const context = conversationContexts.get(sessionId);
    context.lastActivity = now;
    
    return context;
}

/**
 * Clean up expired conversation sessions
 */
function cleanupExpiredSessions(now) {
    const expiredSessions = [];
    
    for (const [sessionId, context] of conversationContexts.entries()) {
        if (now - context.lastActivity > SESSION_TIMEOUT) {
            expiredSessions.push(sessionId);
        }
    }
    
    expiredSessions.forEach(sessionId => {
        conversationContexts.delete(sessionId);
        console.log(`Cleaned up expired session: ${sessionId}`);
    });
}

/**
 * Update conversation context with new message and response
 */
function updateConversationContext(context, userMessage, response, queryAnalysis) {
    // Add message to history
    context.messages.push({
        type: 'user',
        content: userMessage,
        timestamp: Date.now(),
        intent: queryAnalysis.intent,
        entities: queryAnalysis.entities
    });
    
    context.messages.push({
        type: 'assistant',
        content: response.text,
        timestamp: Date.now(),
        confidence: response.confidence,
        sources: response.sources
    });
    
    // Keep only last N messages
    if (context.messages.length > MAX_CONTEXT_MESSAGES * 2) {
        context.messages = context.messages.slice(-MAX_CONTEXT_MESSAGES * 2);
    }
    
    // Update entities
    queryAnalysis.entities.forEach(entity => {
        if (!context.entities.has(entity.type)) {
            context.entities.set(entity.type, new Set());
        }
        context.entities.get(entity.type).add(entity.value);
    });
    
    // Update intent history
    context.intentHistory.push({
        intent: queryAnalysis.intent,
        confidence: queryAnalysis.intentConfidence,
        timestamp: Date.now()
    });
    
    // Update response statistics
    context.totalResponses++;
    if (response.confidence > CONFIG.confidenceThreshold) {
        context.successfulResponses++;
    }
    
    context.lastActivity = Date.now();
}

/**
 * Generate a session ID
 */
function generateSessionId() {
    return 'sess_' + Math.random().toString(36).substr(2, 12) + '_' + Date.now();
}

/**
 * Ensure knowledge base is loaded and up-to-date
 */
async function ensureKnowledgeBase() {
    try {
        const knowledgeBasePath = path.join(process.cwd(), 'knowledge_base.txt');
        
        // Check if file exists
        if (!fs.existsSync(knowledgeBasePath)) {
            console.warn('Knowledge base file not found, creating empty knowledge base');
            knowledgeBase = {
                chunks: [],
                embeddings: [],
                metadata: {
                    totalChunks: 0,
                    lastProcessed: Date.now(),
                    version: '1.0'
                }
            };
            // initialize empty MiniSearch index
            await buildMiniSearchIndex([]);
            return;
        }
        
        const stats = fs.statSync(knowledgeBasePath);
        const fileModified = stats.mtime.getTime();
        
        // Check if we need to reload
        if (!knowledgeBase || fileModified > lastProcessed) {
            console.log('Loading/reloading knowledge base...');
            
            const knowledgeText = fs.readFileSync(knowledgeBasePath, 'utf-8');
            knowledgeBase = await processKnowledgeBase(knowledgeText);
            lastProcessed = Date.now();

            // Build/update MiniSearch index from chunks
            await buildMiniSearchIndex(knowledgeBase.chunks);

            console.log(`Knowledge base loaded: ${knowledgeBase.chunks.length} chunks; miniSearchReady=${miniSearchReady}`);
        }
        
    } catch (error) {
        console.error('Error loading knowledge base:', error);
        // Continue with empty knowledge base
        knowledgeBase = {
            chunks: [],
            embeddings: [],
            metadata: {
                totalChunks: 0,
                lastProcessed: Date.now(),
                version: '1.0'
            }
        };
        await buildMiniSearchIndex([]);
    }
}

/**
 * Process knowledge base text into chunks and embeddings
 */
async function processKnowledgeBase(knowledgeText) {
    try {
        // Split text into chunks
        const chunks = splitIntoChunks(knowledgeText);
        
        // Generate embeddings for each chunk
        const embeddings = [];
        const processedChunks = [];
        
        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            if (chunk.trim().length === 0) continue;
            
            try {
                const embedding = await generateEmbedding(chunk);
                
                processedChunks.push({
                    id: generateId(),
                    text: chunk,
                    embedding: embedding,
                    metadata: {
                        source: 'knowledge_base.txt',
                        index: i,
                        lastUpdated: Date.now()
                    }
                });
                
                embeddings.push(embedding);
                
            } catch (error) {
                console.error(`Error processing chunk ${i}:`, error);
                continue;
            }
        }
        
        return {
            chunks: processedChunks,
            embeddings: embeddings,
            metadata: {
                totalChunks: processedChunks.length,
                lastProcessed: Date.now(),
                version: '1.0'
            }
        };
        
    } catch (error) {
        console.error('Error processing knowledge base:', error);
        throw error;
    }
}

/**
 * Split text into chunks
 */
function splitIntoChunks(text) {
    // Split by double newlines (paragraph separation)
    const paragraphs = text.split(/\\n\\n/).filter(p => p.trim().length > 0);
    
    const chunks = [];
    
    for (const paragraph of paragraphs) {
        if (paragraph.length <= CONFIG.chunkSize) {
            chunks.push(paragraph.trim());
        } else {
            // Split long paragraphs into smaller chunks
            const sentences = paragraph.split(/[.!?]+/).filter(s => s.trim().length > 0);
            let currentChunk = '';
            
            for (const sentence of sentences) {
                const trimmedSentence = sentence.trim();
                if (currentChunk.length + trimmedSentence.length + 1 <= CONFIG.chunkSize) {
                    currentChunk += (currentChunk ? '. ' : '') + trimmedSentence;
                } else {
                    if (currentChunk) {
                        chunks.push(currentChunk + '.');
                    }
                    currentChunk = trimmedSentence;
                }
            }
            
            if (currentChunk) {
                chunks.push(currentChunk + '.');
            }
        }
    }
    
    return chunks;
}

/**
 * Generate embedding for text (legacy function - now uses cached version)
 */
async function generateEmbedding(text) {
    return await generateEmbeddingWithCache(text);
}

/**
 * Advanced multi-stage semantic search with hybrid retrieval
 */
function searchSimilarChunks(queryEmbedding, topK = 5, query = '', conversationContext = null) {
    if (!knowledgeBase || knowledgeBase.chunks.length === 0) {
        return [];
    }
    
    // Stage 1: Semantic similarity search
    const semanticResults = performSemanticSearch(queryEmbedding, topK * 3);
    
    // Stage 2: Keyword-based search (MiniSearch BM25) for exact/rare terms
    const keywordResults = performKeywordSearch(query, topK * 2);
    
    // Stage 3: Context-aware search using conversation history
    const contextResults = performContextualSearch(queryEmbedding, conversationContext, topK * 2);
    
    // Stage 4: Combine and rerank all results
    const combinedResults = combineAndRerankResults({
        semantic: semanticResults,
        keyword: keywordResults,
        contextual: contextResults
    }, query, conversationContext, topK);
    
    return combinedResults;
}

/**
 * Stage 1: Enhanced semantic search with multiple similarity metrics
 */
function performSemanticSearch(queryEmbedding, topK) {
    const results = knowledgeBase.chunks.map(chunk => {
        const cosine = cosineSimilarity(queryEmbedding, chunk.embedding);
        const euclidean = 1 / (1 + euclideanDistance(queryEmbedding, chunk.embedding));
        const manhattan = 1 / (1 + manhattanDistance(queryEmbedding, chunk.embedding));
        
        // Weighted combination of similarity metrics
        const combinedSimilarity = (cosine * 0.7) + (euclidean * 0.2) + (manhattan * 0.1);
        
        return {
            chunk,
            similarity: combinedSimilarity,
            metadata: { ...chunk.metadata, searchType: 'semantic' },
            scores: { cosine, euclidean, manhattan, combined: combinedSimilarity }
        };
    });
    
    return results
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, topK)
        .filter(result => result.similarity > 0.15);
}

/**
 * Stage 2: Keyword-based search for exact matches
 */
function performKeywordSearch(query, topK) {
    if (!query || typeof query !== 'string') {
        return [];
    }
    if (!CONFIG.hybrid.enableMiniSearch || !miniSearchReady || !miniSearch) {
        // fallback to naive keyword scoring if MiniSearch is not ready
        const queryWords = query.toLowerCase()
            .replace(/[^\w\s]/g, ' ')
            .split(/\s+/)
            .filter(word => word.length > 2);
        if (queryWords.length === 0) return [];
        const results = knowledgeBase.chunks.map(chunk => {
            const chunkText = chunk.text.toLowerCase();
            let score = 0, exactMatches = 0, partialMatches = 0;
            queryWords.forEach(word => {
                const exactCount = (chunkText.match(new RegExp(`\\b${word}\\b`, 'g')) || []).length;
                if (exactCount > 0) { exactMatches++; score += exactCount * 2; }
                if (chunkText.includes(word)) { partialMatches++; score += 0.5; }
            });
            const normalizedScore = score / (queryWords.length * 2);
            return {
                chunk,
                similarity: normalizedScore,
                metadata: { ...chunk.metadata, searchType: 'keyword', exactMatches, partialMatches }
            };
        });
        return results
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, topK)
            .filter(result => result.similarity > 0.1);
    }

    // MiniSearch path
    try {
        const msResults = miniSearch.search(query, {
            prefix: true, // allow partials for rare terms
            fuzzy: 0,     // keep strict unless you want limited fuzziness
            boost: { text: 1 }
        });
        // Normalize MiniSearch score to 0..1 by dividing by max score observed
        const maxScore = msResults.length > 0 ? msResults[0].score : 1;
        const mapped = msResults.slice(0, topK).map(r => {
            const chunk = knowledgeBase.chunks.find(c => c.id === r.id) || { text: r.text || '', metadata: {} };
            return {
                chunk,
                similarity: maxScore ? (r.score / maxScore) : 0,
                metadata: { ...chunk.metadata, searchType: 'keyword', minisearchScore: r.score }
            };
        });
        return mapped;
    } catch (e) {
        console.warn('MiniSearch search failed, falling back:', e?.message);
        return [];
    }
}

/**
 * Stage 3: Context-aware search using conversation history
 */
function performContextualSearch(queryEmbedding, conversationContext, topK) {
    if (!conversationContext || !conversationContext.entities) {
        return [];
    }
    // entities is stored as a Map of type -> Set(values). Fallback to array shape if ever different.
    const entitiesIterable = conversationContext.entities instanceof Map
        ? Array.from(conversationContext.entities.entries()).flatMap(([type, set]) =>
            Array.from(set || []).map(value => ({ type, value })))
        : Array.isArray(conversationContext.entities)
            ? conversationContext.entities
            : [];

    if (entitiesIterable.length === 0) {
        return [];
    }

    const results = knowledgeBase.chunks.map(chunk => {
        let contextScore = 0;
        let entityMatches = 0;

        entitiesIterable.forEach(entity => {
            const safe = (entity.value || '').toString().replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            if (!safe) return;
            const entityRegex = new RegExp(safe, 'gi');
            const matches = (chunk.text.match(entityRegex) || []).length;

            if (matches > 0) {
                entityMatches++;
                // Weight by entity type importance
                const typeWeight = getEntityTypeWeight(entity.type?.toUpperCase?.() || entity.type || 'OTHER');
                contextScore += matches * typeWeight;
            }
        });

        // Boost for chunks that match multiple entities
        if (entityMatches > 1) {
            contextScore *= (1 + entityMatches * 0.3);
        }

        const normalizedScore = Math.min(contextScore / Math.max(1, entitiesIterable.length), 1.0);

        return {
            chunk,
            similarity: normalizedScore,
            metadata: {
                ...chunk.metadata,
                searchType: 'contextual',
                entityMatches,
                contextScore
            }
        };
    });

    return results
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, topK)
        .filter(result => result.similarity > 0.05);
}

/**
 * Stage 4: Combine and rerank results from all search stages
 */
function combineAndRerankResults(results, query, conversationContext, topK) {
    const { semantic, keyword, contextual } = results;
    const combinedMap = new Map();
    
    const weights = CONFIG.hybrid || { semanticWeight: 0.6, keywordWeight: 0.35, contextualWeight: 0.1, multiTypeBoost: 1.2 };

    // Combine results with weighted scoring
    const processResults = (resultList, weight, boost = 1) => {
        (resultList || []).forEach(result => {
            const chunkId = result.chunk.id;
            const weightedScore = (result.similarity || 0) * weight * boost;
            
            if (combinedMap.has(chunkId)) {
                const existing = combinedMap.get(chunkId);
                existing.combinedScore += weightedScore;
                existing.searchTypes.push(result.metadata.searchType);
                existing.allScores[result.metadata.searchType] = result.similarity;
            } else {
                combinedMap.set(chunkId, {
                    ...result,
                    combinedScore: weightedScore,
                    searchTypes: [result.metadata.searchType],
                    allScores: { [result.metadata.searchType]: result.similarity }
                });
            }
        });
    };
    
    // Weight different search types (configurable)
    processResults(semantic, weights.semanticWeight);
    processResults(keyword, weights.keywordWeight, 1.5);
    processResults(contextual, weights.contextualWeight, 2.0);
    
    // Convert map to array and apply final reranking
    const finalResults = Array.from(combinedMap.values())
        .map(result => {
            // Boost for multiple search type matches
            const searchTypeBoost = result.searchTypes.length > 1 ? weights.multiTypeBoost : 1.0;
            
            // Boost for recent chunks (if timestamp available)
            const freshnessBoost = calculateFreshnessBoost(result.chunk.metadata);
            
            // Apply final scoring
            result.finalScore = result.combinedScore * searchTypeBoost * freshnessBoost;
            
            return result;
        })
        .sort((a, b) => b.finalScore - a.finalScore)
        .slice(0, topK);
    
    return finalResults;
}

/**
 * Get weight for different entity types
 */
function getEntityTypeWeight(entityType) {
    const weights = {
        'BUSINESS_INFO': 1.5,
        'CONTACT': 1.4,
        'PRODUCT': 1.3,
        'PRICE': 1.2,
        'POLICY': 1.1,
        'PERSON': 1.0,
        'LOCATION': 0.9,
        'DATE': 0.8,
        'OTHER': 0.5
    };
    return weights[entityType] || 0.5;
}

/**
 * Calculate freshness boost based on chunk metadata
 */
function calculateFreshnessBoost(metadata) {
    if (!metadata || !metadata.lastUpdated) {
        return 1.0;
    }
    
    const daysSinceUpdate = (Date.now() - metadata.lastUpdated) / (1000 * 60 * 60 * 24);
    
    // Boost recent content, but don't penalize older content too much
    if (daysSinceUpdate <= 7) return 1.1;
    if (daysSinceUpdate <= 30) return 1.05;
    if (daysSinceUpdate <= 90) return 1.0;
    return 0.95;
}

/**
 * Enhanced similarity metrics
 */
function euclideanDistance(vectorA, vectorB) {
    if (!vectorA || !vectorB || vectorA.length !== vectorB.length) {
        return Infinity;
    }
    
    let sum = 0;
    for (let i = 0; i < vectorA.length; i++) {
        const diff = vectorA[i] - vectorB[i];
        sum += diff * diff;
    }
    return Math.sqrt(sum);
}

function manhattanDistance(vectorA, vectorB) {
    if (!vectorA || !vectorB || vectorA.length !== vectorB.length) {
        return Infinity;
    }
    
    let sum = 0;
    for (let i = 0; i < vectorA.length; i++) {
        sum += Math.abs(vectorA[i] - vectorB[i]);
    }
    return sum;
}

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(vectorA, vectorB) {
    if (!vectorA || !vectorB || vectorA.length !== vectorB.length) {
        return 0;
    }
    
    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;
    
    for (let i = 0; i < vectorA.length; i++) {
        dotProduct += vectorA[i] * vectorB[i];
        magnitudeA += vectorA[i] * vectorA[i];
        magnitudeB += vectorB[i] * vectorB[i];
    }
    
    const magnitude = Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
}

/**
 * Analyze user query with intent classification and entity extraction
 */
async function analyzeQuery(message, conversationContext, requestId) {
    try {
        const analysis = {
            originalQuery: message,
            intent: 'unknown',
            intentConfidence: 0,
            entities: [],
            expandedQueries: [],
            contextualReferences: [],
            isFollowUp: false
        };
        
        // Intent classification
        const intentResult = await classifyIntent(message, conversationContext);
        analysis.intent = intentResult.name;
        analysis.intentConfidence = intentResult.confidence;
        
        // Entity extraction
        analysis.entities = extractEntities(message, conversationContext);
        
        // Check if this is a follow-up question
        analysis.isFollowUp = detectFollowUpQuestion(message, conversationContext);
        
        // Resolve contextual references ("it", "that", "this", etc.)
        analysis.contextualReferences = resolveContextualReferences(message, conversationContext);
        
        // Query expansion for better retrieval
        if (CONFIG.queryExpansionEnabled) {
            analysis.expandedQueries = await expandQuery(message, analysis, conversationContext);
        }
        
        console.log(`[${requestId}] Query analysis completed:`, {
            intent: analysis.intent,
            confidence: analysis.intentConfidence,
            isFollowUp: analysis.isFollowUp,
            entities: analysis.entities.length,
            expansions: analysis.expandedQueries.length
        });
        
        return analysis;
        
    } catch (error) {
        console.error(`[${requestId}] Query analysis error:`, error);
        // Return basic analysis on error
        return {
            originalQuery: message,
            intent: 'question',
            intentConfidence: 0.5,
            entities: [],
            expandedQueries: [message],
            contextualReferences: [],
            isFollowUp: false
        };
    }
}

/**
 * Classify the intent of the user message
 */
async function classifyIntent(message, conversationContext) {
    const intents = [
        { name: 'question', keywords: ['what', 'how', 'why', 'when', 'where', 'who', '?'], weight: 1.0 },
        { name: 'request', keywords: ['can you', 'please', 'help me', 'i need', 'show me'], weight: 1.0 },
        { name: 'clarification', keywords: ['explain', 'clarify', 'what do you mean', 'more details'], weight: 0.9 },
        { name: 'confirmation', keywords: ['yes', 'no', 'correct', 'right', 'wrong', 'exactly'], weight: 0.8 },
        { name: 'greeting', keywords: ['hello', 'hi', 'hey', 'good morning', 'good afternoon'], weight: 0.7 },
        { name: 'complaint', keywords: ['problem', 'issue', 'error', 'not working', 'broken'], weight: 0.9 },
        { name: 'compliment', keywords: ['thank you', 'thanks', 'great', 'awesome', 'helpful'], weight: 0.6 }
    ];
    
    const messageLower = message.toLowerCase();
    let bestIntent = { name: 'question', confidence: 0.5 };
    let maxScore = 0;
    
    for (const intent of intents) {
        let score = 0;
        for (const keyword of intent.keywords) {
            if (messageLower.includes(keyword)) {
                score += intent.weight;
            }
        }
        
        // Boost score if this intent has been used recently
        const recentIntents = conversationContext.intentHistory.slice(-3);
        const recentIntentCount = recentIntents.filter(h => h.intent === intent.name).length;
        score += recentIntentCount * 0.1;
        
        if (score > maxScore) {
            maxScore = score;
            bestIntent = {
                name: intent.name,
                confidence: Math.min(score / Math.max(intent.keywords.length, 1), 1.0)
            };
        }
    }
    
    return bestIntent;
}

/**
 * Extract entities from the message
 */
function extractEntities(message, conversationContext) {
    const entities = [];
    const messageLower = message.toLowerCase();
    
    // Business hours patterns
    const timePatterns = [
        { regex: /\b([0-9]{1,2}):?([0-9]{2})\s*(am|pm)\b/gi, type: 'time' },
        { regex: /\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/gi, type: 'day' },
        { regex: /\b(hours?|schedule|time)\b/gi, type: 'business_info' }
    ];
    
    // Contact information patterns
    const contactPatterns = [
        { regex: /\b(phone|call|contact|email|support)\b/gi, type: 'contact' },
        { regex: /\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b/gi, type: 'email' },
        { regex: /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/gi, type: 'phone' }
    ];
    
    // Product/service patterns
    const productPatterns = [
        { regex: /\b(product|service|plan|pricing|cost|price)\b/gi, type: 'product' },
        { regex: /\$[0-9]+(\.\d{2})?\b/gi, type: 'price' }
    ];
    
    const allPatterns = [...timePatterns, ...contactPatterns, ...productPatterns];
    
    allPatterns.forEach(pattern => {
        let match;
        while ((match = pattern.regex.exec(message)) !== null) {
            entities.push({
                type: pattern.type,
                value: match[0],
                position: match.index
            });
            
            // Prevent infinite loop
            if (!pattern.regex.global) break;
        }
    });
    
    return entities;
}

/**
 * Detect if this is a follow-up question
 */
function detectFollowUpQuestion(message, conversationContext) {
    if (conversationContext.messages.length === 0) {
        return false;
    }
    
    const followUpIndicators = [
        'also', 'what about', 'and', 'additionally', 'furthermore',
        'can you also', 'what else', 'more information', 'tell me more'
    ];
    
    const messageLower = message.toLowerCase();
    return followUpIndicators.some(indicator => messageLower.includes(indicator));
}

/**
 * Resolve contextual references in the message
 */
function resolveContextualReferences(message, conversationContext) {
    const references = [];
    const messageLower = message.toLowerCase();
    
    const pronouns = ['it', 'that', 'this', 'they', 'them', 'those', 'these'];
    
    pronouns.forEach(pronoun => {
        if (messageLower.includes(pronoun)) {
            // Get the last assistant message for context
            const lastMessages = conversationContext.messages.slice(-4);
            const lastAssistantMessage = lastMessages.reverse().find(msg => msg.type === 'assistant');
            
            if (lastAssistantMessage) {
                references.push({
                    pronoun,
                    referenceContext: lastAssistantMessage.content.substring(0, 100)
                });
            }
        }
    });
    
    return references;
}

/**
 * Expand query for better retrieval
 */
async function expandQuery(originalQuery, analysis, conversationContext) {
    const expandedQueries = [originalQuery];
    
    try {
        // Add context from recent conversation
        if (conversationContext.messages.length > 0) {
            const recentContext = conversationContext.messages
                .slice(-4)
                .filter(msg => msg.type === 'user')
                .map(msg => msg.content)
                .join(' ');
            
            if (recentContext && recentContext !== originalQuery) {
                expandedQueries.push(`${recentContext} ${originalQuery}`);
            }
        }
        
        // Add entity-based expansions
        analysis.entities.forEach(entity => {
            if (entity.type === 'business_info') {
                expandedQueries.push(`business information ${originalQuery}`);
            } else if (entity.type === 'contact') {
                expandedQueries.push(`contact support ${originalQuery}`);
            } else if (entity.type === 'product') {
                expandedQueries.push(`product information ${originalQuery}`);
            }
        });
        
        // Intent-based expansion
        if (analysis.intent === 'question') {
            expandedQueries.push(`FAQ ${originalQuery}`);
        } else if (analysis.intent === 'request') {
            expandedQueries.push(`how to ${originalQuery}`);
        }
        
        // Limit expansions
        return expandedQueries.slice(0, CONFIG.maxQueryExpansions + 1);
        
    } catch (error) {
        console.error('Query expansion error:', error);
        return [originalQuery];
    }
}

/**
 * Context-aware retrieval with enhanced scoring
 */
async function performContextAwareRetrieval(queryAnalysis, conversationContext, requestId) {
    try {
        const chunkScores = new Map();
        
        // Process each expanded query
        for (const query of queryAnalysis.expandedQueries) {
            const queryEmbedding = await generateEmbedding(query);
            const chunks = searchSimilarChunks(queryEmbedding, CONFIG.maxRetrievalChunks * 3, query, conversationContext);
            
            chunks.forEach(chunk => {
                const chunkId = chunk.chunk.id;
                const baseScore = chunk.similarity;
                
                // Context boost based on conversation history
                let contextBoost = 0;
                if (conversationContext.messages.length > 0) {
                    const recentTopics = extractTopicsFromHistory(conversationContext);
                    contextBoost = calculateContextBoost(chunk.chunk.text, recentTopics);
                }
                
                // Intent alignment boost
                const intentBoost = calculateIntentBoost(chunk.chunk.text, queryAnalysis.intent);
                
                // Entity alignment boost
                const entityBoost = calculateEntityBoost(chunk.chunk.text, queryAnalysis.entities);
                
                const finalScore = baseScore + (contextBoost * CONFIG.contextWeight) + intentBoost + entityBoost;
                
                if (!chunkScores.has(chunkId) || chunkScores.get(chunkId).score < finalScore) {
                    chunkScores.set(chunkId, {
                        chunk: chunk.chunk,
                        score: finalScore,
                        baseScore,
                        contextBoost,
                        intentBoost,
                        entityBoost,
                        metadata: chunk.metadata
                    });
                }
            });
        }
        
        // Sort and take more to allow sentence packing selection
        const topChunks = Array.from(chunkScores.values())
            .sort((a, b) => b.score - a.score)
            .slice(0, CONFIG.maxRetrievalChunks * 2);

        // Sentence-level packing
        let packed = topChunks;
        if (CONFIG.packing.enableSentencePacking) {
            packed = packSentencesFromChunks(topChunks, CONFIG.packing.maxContextTokens);
        }
        
        const result = {
            chunks: packed,
            bestSimilarity: packed.length > 0 ? packed[0].baseScore : 0,
            contextBoost: packed.length > 0 ? packed[0].contextBoost : 0,
            averageScore: packed.length > 0 ?
                packed.reduce((sum, c) => sum + (c.score || 0), 0) / packed.length : 0
        };
        
        console.log(`[${requestId}] Context-aware retrieval completed:`, {
            totalChunksEvaluated: chunkScores.size,
            finalChunksSelected: packed.length,
            bestScore: result.bestSimilarity,
            averageScore: result.averageScore,
            sentencePacked: CONFIG.packing.enableSentencePacking
        });
        
        return result;
        
    } catch (error) {
        console.error(`[${requestId}] Context-aware retrieval error:`, error);
        // Fallback to basic retrieval
        const queryEmbedding = await generateEmbedding(queryAnalysis.originalQuery);
        const chunks = searchSimilarChunks(queryEmbedding, CONFIG.maxRetrievalChunks, queryAnalysis.originalQuery, conversationContext);
        const packed = CONFIG.packing.enableSentencePacking ? packSentencesFromChunks(chunks.map(c => ({
            chunk: c.chunk,
            score: c.similarity,
            baseScore: c.similarity,
            contextBoost: 0,
            intentBoost: 0,
            entityBoost: 0,
            metadata: c.metadata
        })), CONFIG.packing.maxContextTokens) : chunks;
        return {
            chunks: packed,
            bestSimilarity: packed.length > 0 ? (packed[0].baseScore || 0) : 0,
            contextBoost: 0,
            averageScore: packed.length > 0 ?
                packed.reduce((sum, c) => sum + (c.score || 0), 0) / packed.length : 0
        };
    }
}

/**
 * Extract topics from conversation history
 */
function extractTopicsFromHistory(conversationContext) {
    const topics = new Set();
    
    conversationContext.messages.forEach(msg => {
        if (msg.type === 'user' && msg.entities) {
            msg.entities.forEach(entity => {
                topics.add(entity.type);
            });
        }
    });
    
    return Array.from(topics);
}

/**
 * Calculate context boost based on conversation topics
 */
function calculateContextBoost(chunkText, recentTopics) {
    let boost = 0;
    const chunkLower = chunkText.toLowerCase();
    
    recentTopics.forEach(topic => {
        const topicKeywords = getTopicKeywords(topic);
        topicKeywords.forEach(keyword => {
            if (chunkLower.includes(keyword)) {
                boost += 0.1;
            }
        });
    });
    
    return Math.min(boost, 0.3); // Cap boost at 0.3
}

/**
 * Get keywords for a topic
 */
function getTopicKeywords(topic) {
    const keywordMap = {
        'business_info': ['hours', 'contact', 'business', 'support'],
        'contact': ['email', 'phone', 'call', 'contact', 'support'],
        'product': ['product', 'service', 'pricing', 'plan', 'offer'],
        'time': ['hours', 'schedule', 'time', 'open', 'closed'],
        'day': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    };
    
    return keywordMap[topic] || [];
}

/**
 * Calculate intent alignment boost
 */
function calculateIntentBoost(chunkText, intent) {
    const intentKeywords = {
        'question': ['what', 'how', 'when', 'where', 'why'],
        'request': ['help', 'can', 'please', 'need'],
        'complaint': ['problem', 'issue', 'error', 'not working'],
        'greeting': ['welcome', 'hello', 'help']
    };
    
    const keywords = intentKeywords[intent] || [];
    const chunkLower = chunkText.toLowerCase();
    
    let boost = 0;
    keywords.forEach(keyword => {
        if (chunkLower.includes(keyword)) {
            boost += 0.05;
        }
    });
    
    return Math.min(boost, 0.2); // Cap boost at 0.2
}

/**
 * Calculate entity alignment boost
 */
function calculateEntityBoost(chunkText, entities) {
    let boost = 0;
    const chunkLower = chunkText.toLowerCase();
    
    entities.forEach(entity => {
        if (chunkLower.includes(entity.value.toLowerCase())) {
            boost += 0.1;
        }
    });
    
    return Math.min(boost, 0.25); // Cap boost at 0.25
}

/**
 * Generate multiple candidate responses
 */
async function generateCandidateResponses(queryAnalysis, retrievalResults, conversationContext, requestId) {
    try {
        const candidates = [];
        
        // Generate primary response with full context
        const primaryCandidate = await generateContextAwareResponse(
            queryAnalysis, 
            retrievalResults, 
            conversationContext, 
            'comprehensive'
        );
        candidates.push({ ...primaryCandidate, type: 'comprehensive' });
        
        // Generate concise response if confidence is high
        if (retrievalResults.bestSimilarity > 0.7) {
            const conciseCandidate = await generateContextAwareResponse(
                queryAnalysis, 
                retrievalResults, 
                conversationContext, 
                'concise'
            );
            candidates.push({ ...conciseCandidate, type: 'concise' });
        }
        
        // Generate explanatory response for complex queries
        if (queryAnalysis.entities.length > 2 || queryAnalysis.isFollowUp) {
            const explanatoryCandidate = await generateContextAwareResponse(
                queryAnalysis, 
                retrievalResults, 
                conversationContext, 
                'explanatory'
            );
            candidates.push({ ...explanatoryCandidate, type: 'explanatory' });
        }
        
        console.log(`[${requestId}] Generated ${candidates.length} candidate responses`);
        return candidates;
        
    } catch (error) {
        console.error(`[${requestId}] Candidate generation error:`, error);
        // Fallback to single response
        const fallbackResponse = await generateResponse(queryAnalysis.originalQuery, 
            retrievalResults.chunks.map(chunk => ({ chunk: chunk.chunk, similarity: chunk.score })));
        return [{ 
            text: fallbackResponse, 
            confidence: retrievalResults.bestSimilarity,
            type: 'fallback'
        }];
    }
}

/**
 * Generate context-aware response with dynamic prompting
 */
async function generateContextAwareResponse(queryAnalysis, retrievalResults, conversationContext, responseType) {
    try {
        const context = retrievalResults.chunks
            .map(result => `- ${result.chunk.text}`)
            .join('\n');
        
        // Build conversation history context
        const conversationHistory = conversationContext.messages
            .slice(-6) // Last 3 exchanges
            .map(msg => `${msg.type === 'user' ? 'User' : 'Assistant'}: ${msg.content}`)
            .join('\n');
        
        // Dynamic prompt based on response type and context
        const prompt = await buildDynamicPrompt(
            queryAnalysis,
            context,
            conversationHistory,
            responseType,
            conversationContext
        );
        
        const generationStartTime = Date.now();
        const result = await chatModel.generateContent({
            contents: [{ role: 'user', parts: [{ text: prompt }] }],
            generationConfig: {
                maxOutputTokens: getMaxTokensForResponseType(responseType),
                temperature: getTemperatureForResponseType(responseType),
                topP: 0.8,
                topK: 40
            }
        });
        
        performanceMonitor.recordGeneration(Date.now() - generationStartTime);
        
        const responseText = result.response.text();

        // Pre-sanitize before validation to reduce meta leakage
        const preSanitized = sanitizeResponseText(responseText);

        // Post-generation validator with single revision pass
        const validated = await runValidatorAndMaybeRevise(preSanitized, {
            personaEnabled: (conversationContext?._requestPersona === 'senior_support') || CONFIG.supportPersonaEnabled
        });

        // Calculate confidence based on multiple factors
        const confidence = calculateResponseConfidence(retrievalResults, queryAnalysis, validated.text);

        // Final sanitize in case revision reintroduced any headings
        const finalText = sanitizeResponseText(validated.text);

        return {
            text: finalText,
            confidence,
            sources: retrievalResults.chunks.map(chunk => chunk.metadata?.source || 'knowledge_base').filter(Boolean),
            validation: validated.meta
        };
        
    } catch (error) {
        console.error('Context-aware response generation error:', error);
        throw error;
    }
}

/**
 * Build dynamic prompt based on context and response type
 */
async function buildDynamicPrompt(queryAnalysis, context, conversationHistory, responseType, conversationContext) {
    // Determine persona enablement:
    // 1) Per-request override via req.body.persona === 'senior_support'
    // 2) Fallback to env flag SUPPORT_PERSONA
    const enableSupportPersona = (conversationContext?._requestPersona === 'senior_support') || CONFIG.supportPersonaEnabled;

    let sections = [];

    // Optional persona block
    if (enableSupportPersona) {
        sections.push(buildSeniorSupportPersonaBlock());
    }

    // Dynamic few-shot selection from seeds (if persona on)
    if (enableSupportPersona && CONFIG.fewshot.enable && supportSeeds && supportSeeds.length > 0) {
        try {
            const k = Math.max(1, CONFIG.fewshot.k || 3);
            const selected = await selectFewShotSeeds(queryAnalysis.originalQuery, supportSeeds, k, CONFIG.fewshot.minSimilarity);
            const fewShotStr = renderFewShotExamples(selected.map(s => ({ user: s.user, assistant: s.assistant })));
            if (fewShotStr) {
                sections.push(fewShotStr);
            }
        } catch (e) {
            console.warn('Few-shot selection failed:', e?.message);
        }
    }

    // Base assistant identity with clear style guidance to avoid checklist/plan/summary boilerplate
    sections.push([
        'You are RagZzy, a helpful customer support assistant.',
        'Respond conversationally and directly to the user’s question.',
        'Do not include meta sections like "Summary:", "Steps:", "Validation:", "Rollback:", or "Notes:".',
        'Do not list generic capabilities or describe internal processes unless explicitly asked.',
        'Avoid bullet-point checklists unless the user asks for steps or a list.',
        'Prefer a concise, helpful paragraph or two that answers the question.'
    ].join(' '));

    // Conversation history
    if (conversationHistory) {
        sections.push(`Previous conversation:\n${conversationHistory}`);
    }

    // Retrieved context
    if (context) {
        sections.push(`Relevant information:\n${context}`);
    }

    // Current query and contextual refs
    const parts = [`Current question: ${queryAnalysis.originalQuery}`];
    if (queryAnalysis.contextualReferences.length > 0) {
        parts.push(
            `Note: The user used references like "${queryAnalysis.contextualReferences.map(ref => ref.pronoun).join(', ')}" which may refer to previous discussion topics.`
        );
    }
    sections.push(parts.join('\n'));

    // Response type instructions with anti-boilerplate constraint
    const responseInstructions = {
        'comprehensive': 'Provide a detailed, thorough response that fully addresses the question with relevant details and context. Do not include boilerplate sections or headings.',
        'concise': 'Provide a brief, direct answer that gets straight to the point while being helpful. No boilerplate headings.',
        'explanatory': 'Provide a clear explanation with step-by-step details only if steps are explicitly helpful. Avoid meta headings.'
    };
    sections.push(responseInstructions[responseType] || responseInstructions['comprehensive']);

    // Optional personalization (if available in this module)
    try {
        if (typeof getPersonalizedResponseInstructions === 'function' && conversationContext?.sessionId) {
            const personalization = getPersonalizedResponseInstructions(conversationContext.sessionId, queryAnalysis);
            if (personalization) {
                sections.push(`Personalization: ${personalization}`);
            }
        }
    } catch (_e) {
        // ignore personalization errors
    }

    // Intent-specific guidance
    if (queryAnalysis.intent === 'clarification') {
        sections.push('The user is asking for clarification, so be extra clear and specific in your explanation.');
    } else if (queryAnalysis.intent === 'complaint') {
        sections.push('The user seems to have an issue or concern. Be empathetic and solution-focused.');
    } else if (queryAnalysis.isFollowUp) {
        sections.push('This appears to be a follow-up question, so connect your response to the previous conversation.');
    }

    // Tone reinforcement based on recent success
    if (conversationContext.successfulResponses > 2) {
        sections.push('Maintain the same helpful, effective tone that worked well previously.');
    }

    // Few-shot examples are injected dynamically above when persona is enabled.
    // Static block removed to avoid duplication and token bloat.

    // Final cue
    sections.push('Response:');

    return sections.join('\n\n');
}

/**
 * Get max tokens based on response type
 */
function getMaxTokensForResponseType(responseType) {
    const tokenLimits = {
        'concise': 300,
        'comprehensive': 800,
        'explanatory': 600
    };
    
    return tokenLimits[responseType] || CONFIG.maxResponseTokens;
}

/**
 * Basic safety guard: blocks obvious prompt-injection and disallowed content patterns.
 * This is a lightweight layer; for production, add a full moderation pass.
 */
function basicSafetyGuard(text) {
    const lower = text.toLowerCase();

    // Simple prompt-injection keywords
    const injection = [
        'ignore previous', 'disregard previous', 'system prompt',
        'developer instructions', 'reveal your prompt', 'print your rules'
    ];

    if (injection.some(k => lower.includes(k))) {
        return { allowed: false, reason: 'possible_prompt_injection' };
    }

    // Extremely unsafe content quick checks (expand as needed)
    const disallowed = [
        /child\s*sexual\s*content/i,
        /credit\s*card\s*number/i,
        /social\s*security\s*number/i
    ];
    if (disallowed.some(rx => rx.test(text))) {
        return { allowed: false, reason: 'disallowed_content' };
    }

    // Very long repeated characters
    if (/(.)\1{20,}/.test(text)) {
        return { allowed: false, reason: 'spam' };
    }

    return { allowed: true };
}

/**
 * Get temperature based on response type
 */
function getTemperatureForResponseType(responseType) {
    const temperatures = {
        'concise': 0.3,
        'comprehensive': 0.7,
        'explanatory': 0.5
    };
    
    return temperatures[responseType] || 0.7;
}

/**
 * Calculate response confidence based on multiple factors
 */
function calculateResponseConfidence(retrievalResults, queryAnalysis, responseText) {
    let confidence = retrievalResults.bestSimilarity;
    
    // Boost confidence for high intent confidence
    if (queryAnalysis.intentConfidence > 0.8) {
        confidence += 0.1;
    }
    
    // Boost confidence if multiple chunks contribute
    if (retrievalResults.chunks.length > 2) {
        confidence += 0.05;
    }
    
    // Reduce confidence for very short responses (may indicate lack of info)
    if (responseText.length < 50) {
        confidence -= 0.1;
    }
    
    // Boost confidence if response contains specific information
    const specificityIndicators = ['contact', 'email', 'phone', 'hours', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday'];
    const hasSpecificInfo = specificityIndicators.some(indicator => 
        responseText.toLowerCase().includes(indicator)
    );
    
    if (hasSpecificInfo) {
        confidence += 0.1;
    }
    
    return Math.min(Math.max(confidence, 0), 1); // Clamp between 0 and 1
}

/**
 * Strip residual boilerplate/meta headings the model may emit.
 * Keeps natural language intact; removes leading heading tokens and common section labels.
 */
function sanitizeResponseText(text) {
    if (!text || typeof text !== 'string') return text;
    let t = text;

    // Remove typical section headings at line starts (case-insensitive)
    const headingRx = /^(summary|steps|validation|rollback|notes|plan|task[s]?|checklist)\s*:?\s*/gim;
    t = t.replace(headingRx, '');

    // Remove repeated heading lines
    t = t.split('\n').map(line => line.replace(/^(summary|steps|validation|rollback|notes)\s*:?\s*/i, '')).join('\n');

    // Trim excessive surrounding quotes or code fences the model might introduce
    t = t.replace(/^```[\s\S]*?\n/, '').replace(/```$/,'').trim();

    return t.trim();
}

/**
 * Select the best response from candidates
 */
async function selectBestResponse(candidateResponses, queryAnalysis, conversationContext) {
    if (candidateResponses.length === 1) {
        return {
            ...candidateResponses[0],
            selectionReason: 'only_candidate',
            qualityScore: candidateResponses[0].confidence
        };
    }
    
    let bestCandidate = candidateResponses[0];
    let bestScore = calculateResponseQualityScore(candidateResponses[0], queryAnalysis, conversationContext);
    let selectionReason = 'highest_quality_score';
    
    for (let i = 1; i < candidateResponses.length; i++) {
        const candidate = candidateResponses[i];
        const score = calculateResponseQualityScore(candidate, queryAnalysis, conversationContext);
        
        if (score > bestScore) {
            bestCandidate = candidate;
            bestScore = score;
        }
    }
    
    return {
        ...bestCandidate,
        selectionReason,
        qualityScore: bestScore
    };
}

/**
 * Calculate overall quality score for response selection
 */
function calculateResponseQualityScore(candidate, queryAnalysis, conversationContext) {
    let score = candidate.confidence * 0.4; // Base confidence weight
    
    // Length appropriateness (not too short, not too long)
    const length = candidate.text.length;
    if (length >= 50 && length <= 500) {
        score += 0.2;
    } else if (length > 500) {
        score += 0.1; // Penalize very long responses slightly
    }
    
    // Response type preference based on intent
    if (queryAnalysis.intent === 'question' && candidate.type === 'comprehensive') {
        score += 0.1;
    } else if (queryAnalysis.intent === 'clarification' && candidate.type === 'explanatory') {
        score += 0.1;
    } else if (queryAnalysis.intent === 'greeting' && candidate.type === 'concise') {
        score += 0.1;
    }
    
    // User preference based on history
    const userSuccessRate = conversationContext.totalResponses > 0 ? 
        conversationContext.successfulResponses / conversationContext.totalResponses : 0.5;
    
    if (userSuccessRate > 0.8 && candidate.type === 'comprehensive') {
        score += 0.1; // User likes detailed responses
    }
    
    return Math.min(score, 1.0);
}

/**
 * Generate response using retrieved context (legacy function - kept for backward compatibility)
 */
async function generateResponse(userQuery, retrievedChunks) {
    try {
        const context = retrievedChunks
            .map(result => `- ${result.chunk.text}`)
            .join('\
');
        
        let prompt;
        
        if (retrievedChunks.length === 0) {
            // No relevant context found
            prompt = `You are RagZzy, a helpful customer support assistant. A user asked: '${userQuery}'

I don't have specific information about this topic in my knowledge base. Please provide a brief, helpful response that:
1. Acknowledges that you don't have specific information about their question
2. Suggests they could help by providing the answer to improve the knowledge base
3. Offers to help with other questions you might know about
4. Keeps a friendly, supportive tone

Response:`;
        } else {
            // Use retrieved context
            prompt = `You are RagZzy, a helpful customer support assistant. Based ONLY on the following information, answer the user's question. If the information isn't fully relevant or doesn't contain a complete answer, acknowledge what you do know and mention that additional details could be helpful.

Context:
${context}

User Question: ${userQuery}

Provide a helpful, conversational response based on the context above. If the context doesn't fully answer the question, say so and suggest that the user could help improve the knowledge base.

Response:`;
        }
        
        const result = await chatModel.generateContent({
            contents: [{ role: 'user', parts: [{ text: prompt }] }],
            generationConfig: {
                maxOutputTokens: CONFIG.maxResponseTokens,
                temperature: 0.7,
                topP: 0.8,
                topK: 40
            }
        });
        
        const response = result.response;
        return response.text();
        
    } catch (error) {
        console.error('Error generating response:', error);
        
        // Fallback response
        if (retrievedChunks.length === 0) {
            return "I don't have specific information about that topic yet. Would you like to help by sharing what you know? This would help me provide better answers to future users with similar questions.";
        } else {
            return "I found some related information, but I'm having trouble generating a complete response right now. Please try again, or feel free to contribute additional details to help improve my knowledge base.";
        }
    }
}

/**
 * Get relevant guided prompts based on user query
 */
function getRelevantGuidedPrompts(query) {
    const guidedPrompts = [
        {
            id: 'business-hours',
            question: 'What are your business hours?',
            category: 'business-info',
            keywords: ['hours', 'open', 'closed', 'time', 'schedule']
        },
        {
            id: 'contact-info',
            question: 'How can customers contact support?',
            category: 'business-info',
            keywords: ['contact', 'phone', 'email', 'support', 'help']
        },
        {
            id: 'main-products',
            question: 'What are your main products or services?',
            category: 'products',
            keywords: ['product', 'service', 'offer', 'sell', 'provide']
        },
        {
            id: 'pricing-info',
            question: 'What is your pricing structure?',
            category: 'pricing',
            keywords: ['price', 'cost', 'fee', 'pricing', 'plan', 'subscription']
        },
        {
            id: 'return-policy',
            question: 'What is your return/refund policy?',
            category: 'policies',
            keywords: ['return', 'refund', 'policy', 'exchange', 'money back']
        }
    ];
    
    const queryLower = query.toLowerCase();
    
    // Find prompts with matching keywords
    const relevantPrompts = guidedPrompts.filter(prompt => 
        prompt.keywords.some(keyword => queryLower.includes(keyword))
    );
    
    // Return up to 3 most relevant prompts, or all prompts if none match
    return relevantPrompts.length > 0 ? relevantPrompts.slice(0, 3) : guidedPrompts.slice(0, 3);
}

/**
 * Check if message is spam or inappropriate
 */
function isSpamOrInappropriate(message) {
    const spamPatterns = [
        /buy.{0,10}now/i,
        /click.{0,10}here/i,
        /free.{0,10}money/i,
        /viagra/i,
        /casino/i,
        /\\b(\\w)\\1{4,}/g, // Repeated characters
    ];
    
    return spamPatterns.some(pattern => pattern.test(message));
}

/**
 * Generate a unique request ID
 */
function generateRequestId() {
    return 'req_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
}

/**
 * Generate a unique ID
 */
function generateId() {
    return Math.random().toString(36).substr(2, 9) + '_' + Date.now();
}

/**
 * Categorize and format errors for consistent error responses
 */
function categorizeError(error, requestId, processingTime) {
    console.error(`[${requestId}] Error details:`, {
        message: error.message,
        stack: error.stack?.substring(0, 500),
        type: error.constructor.name
    });
    
    // API-related errors
    if (error.message?.includes('API key') || error.message?.includes('authentication')) {
        return {
            status: 500,
            body: {
                error: 'Service temporarily unavailable due to authentication issues',
                code: 'AUTH_ERROR',
                requestId,
                processingTime,
                retryable: false
            }
        };
    }
    
    // Rate limiting and quota errors
    if (error.message?.includes('quota') || error.message?.includes('rate limit') || error.message?.includes('429')) {
        return {
            status: 429,
            body: {
                error: 'Too many requests. Please try again in a moment.',
                code: 'RATE_LIMITED',
                requestId,
                processingTime,
                retryable: true,
                retryAfter: 60
            }
        };
    }
    
    // Network timeout errors
    if (error.message?.includes('timeout') || error.code === 'ETIMEDOUT') {
        return {
            status: 504,
            body: {
                error: 'Request timed out. Please try again.',
                code: 'TIMEOUT_ERROR',
                requestId,
                processingTime,
                retryable: true
            }
        };
    }
    
    // Memory or resource errors
    if (error.message?.includes('out of memory') || error.message?.includes('resource')) {
        return {
            status: 503,
            body: {
                error: 'Service temporarily overloaded. Please try again.',
                code: 'RESOURCE_ERROR',
                requestId,
                processingTime,
                retryable: true,
                retryAfter: 30
            }
        };
    }
    
    // Validation or input errors
    if (error.message?.includes('invalid') || error.message?.includes('validation')) {
        return {
            status: 400,
            body: {
                error: 'Invalid input provided',
                code: 'VALIDATION_ERROR',
                requestId,
                processingTime,
                retryable: false
            }
        };
    }
    
    // AI model specific errors
    if (error.message?.includes('model') || error.message?.includes('generation')) {
        return {
            status: 502,
            body: {
                error: 'AI service temporarily unavailable',
                code: 'AI_SERVICE_ERROR',
                requestId,
                processingTime,
                retryable: true
            }
        };
    }
    
    // Knowledge base errors
    if (error.message?.includes('knowledge') || error.message?.includes('embedding')) {
        return {
            status: 500,
            body: {
                error: 'Knowledge base temporarily unavailable',
                code: 'KNOWLEDGE_BASE_ERROR',
                requestId,
                processingTime,
                retryable: true
            }
        };
    }
    
    // Generic server errors
    return {
        status: 500,
        body: {
            error: 'An unexpected error occurred. Please try again.',
            code: 'INTERNAL_ERROR',
            requestId,
            processingTime,
            retryable: true
        }
    };
}

/**
 * Performance monitoring and metrics collection
 */
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            requests: {
                total: 0,
                successful: 0,
                failed: 0,
                averageResponseTime: 0
            },
            ai: {
                embeddingCalls: 0,
                generationCalls: 0,
                averageEmbeddingTime: 0,
                averageGenerationTime: 0
            },
            context: {
                activeSessions: 0,
                averageMessagesPerSession: 0,
                contextBoostAverage: 0
            },
            errors: new Map() // error type -> count
        };
        this.responseTimes = [];
        this.embeddingTimes = [];
        this.generationTimes = [];
    }
    
    recordRequest(success, responseTime) {
        this.metrics.requests.total++;
        if (success) {
            this.metrics.requests.successful++;
        } else {
            this.metrics.requests.failed++;
        }
        
        this.responseTimes.push(responseTime);
        if (this.responseTimes.length > 100) {
            this.responseTimes = this.responseTimes.slice(-100);
        }
        
        this.metrics.requests.averageResponseTime = 
            this.responseTimes.reduce((sum, time) => sum + time, 0) / this.responseTimes.length;
    }
    
    recordEmbedding(time) {
        this.metrics.ai.embeddingCalls++;
        this.embeddingTimes.push(time);
        if (this.embeddingTimes.length > 50) {
            this.embeddingTimes = this.embeddingTimes.slice(-50);
        }
        
        this.metrics.ai.averageEmbeddingTime = 
            this.embeddingTimes.reduce((sum, time) => sum + time, 0) / this.embeddingTimes.length;
    }
    
    recordGeneration(time) {
        this.metrics.ai.generationCalls++;
        this.generationTimes.push(time);
        if (this.generationTimes.length > 50) {
            this.generationTimes = this.generationTimes.slice(-50);
        }
        
        this.metrics.ai.averageGenerationTime = 
            this.generationTimes.reduce((sum, time) => sum + time, 0) / this.generationTimes.length;
    }
    
    recordError(errorCode) {
        const count = this.metrics.errors.get(errorCode) || 0;
        this.metrics.errors.set(errorCode, count + 1);
    }
    
    updateContextMetrics() {
        this.metrics.context.activeSessions = conversationContexts.size;
        
        if (conversationContexts.size > 0) {
            const totalMessages = Array.from(conversationContexts.values())
                .reduce((sum, context) => sum + context.messages.length, 0);
            this.metrics.context.averageMessagesPerSession = totalMessages / conversationContexts.size;
        }
    }
    
    getMetrics() {
        this.updateContextMetrics();
        return {
            ...this.metrics,
            errors: Object.fromEntries(this.metrics.errors),
            timestamp: Date.now()
        };
    }
    
    getHealthStatus() {
        const metrics = this.getMetrics();
        const errorRate = metrics.requests.total > 0 ? 
            metrics.requests.failed / metrics.requests.total : 0;
        
        let status = 'healthy';
        const issues = [];
        
        if (errorRate > 0.1) {
            status = 'degraded';
            issues.push(`High error rate: ${(errorRate * 100).toFixed(1)}%`);
        }
        
        if (metrics.requests.averageResponseTime > 10000) {
            status = 'degraded';
            issues.push(`High response time: ${metrics.requests.averageResponseTime.toFixed(0)}ms`);
        }
        
        if (metrics.context.activeSessions > 1000) {
            status = 'warning';
            issues.push(`High session count: ${metrics.context.activeSessions}`);
        }
        
        return {
            status,
            issues,
            metrics
        };
    }
}

// Global performance monitor instance
const performanceMonitor = new PerformanceMonitor();

/**
 * Enhanced embedding generation with caching and performance monitoring
 */
const embeddingCache = new Map();
const EMBEDDING_CACHE_SIZE = 1000;
const EMBEDDING_CACHE_TTL = 60 * 60 * 1000; // 1 hour

async function generateEmbeddingWithCache(text) {
    const startTime = Date.now();
    
    try {
        // Create cache key
        const cacheKey = createCacheKey(text);
        const cached = embeddingCache.get(cacheKey);
        
        if (cached && Date.now() - cached.timestamp < EMBEDDING_CACHE_TTL) {
            performanceMonitor.recordEmbedding(Date.now() - startTime);
            return cached.embedding;
        }
        
        // Generate new embedding
        const result = await embeddingModel.embedContent(text);
        const embedding = result.embedding.values;
        
        // Cache the result
        if (embeddingCache.size >= EMBEDDING_CACHE_SIZE) {
            // Remove oldest entries
            const oldestKey = embeddingCache.keys().next().value;
            embeddingCache.delete(oldestKey);
        }
        
        embeddingCache.set(cacheKey, {
            embedding,
            timestamp: Date.now()
        });
        
        performanceMonitor.recordEmbedding(Date.now() - startTime);
        return embedding;
        
    } catch (error) {
        performanceMonitor.recordEmbedding(Date.now() - startTime);
        console.error('Error generating embedding:', error);
        throw error;
    }
}

/**
 * Lightweight seed embedding cache and few-shot selector
 */
const seedEmbeddingCache = new Map();

async function getSeedEmbedding(seed) {
    const key = seed.id || seed.user.slice(0, 64);
    if (seedEmbeddingCache.has(key)) return seedEmbeddingCache.get(key);
    // Embed on a concatenation of user + assistant to capture topic
    const text = `${seed.user}\n${seed.assistant}`.slice(0, 2000);
    const emb = await generateEmbeddingWithCache(text);
    seedEmbeddingCache.set(key, emb);
    return emb;
}

async function selectFewShotSeeds(query, seeds, k = 3, minSim = 0.2) {
    const qEmb = await generateEmbeddingWithCache(query);
    const scored = [];
    for (const s of seeds) {
        try {
            const e = await getSeedEmbedding(s);
            const sim = cosineSimilarity(qEmb, e);
            scored.push({ seed: s, sim });
        } catch (_e) {}
    }
    scored.sort((a, b) => b.sim - a.sim);
    const picked = scored.filter(x => x.sim >= minSim).slice(0, k).map(x => x.seed);
    // Fallback to top-k even if below minSim to avoid empty block
    return picked.length > 0 ? picked : scored.slice(0, k).map(x => x.seed);
}

/**
 * Simple checklist-based validator with optional single revision.
 */
async function runValidatorAndMaybeRevise(text, { personaEnabled }) {
    const checklist = personaEnabled ? getSupportValidatorChecklist() : [
        'Checklist:',
        '- Answer is helpful and accurate.',
        '- No unsafe content.',
        '- Responds directly to the question.'
    ].join('\n');

    // Ask model to evaluate against checklist and return a JSON verdict
    const judgePrompt = [
        'You are a strict validator. Given the Assistant response below, check the checklist and return a strict JSON object:',
        '{ "ok": boolean, "missing": string[], "notes": string }',
        '',
        checklist,
        '',
        'Assistant response:',
        text
    ].join('\n');

    try {
        const evalRes = await chatModel.generateContent({
            contents: [{ role: 'user', parts: [{ text: judgePrompt }] }],
            generationConfig: { maxOutputTokens: 300, temperature: 0.1, topP: 0.8, topK: 40 }
        });
        const evalText = evalRes.response.text() || '';
        let verdict = { ok: true, missing: [], notes: '' };
        try {
            // Extract JSON substring if model wrapped text
            const jsonMatch = evalText.match(/\{[\s\S]*\}/);
            if (jsonMatch) verdict = JSON.parse(jsonMatch[0]);
        } catch (_e) {
            // If parse fails, consider ok to avoid degradation
            verdict = { ok: true, missing: [], notes: 'parse_failed' };
        }

        if (verdict.ok) {
            return { text, meta: { validated: true, revised: false, verdict } };
        }

        // One revision pass with anti-boilerplate guardrails
        const revisePrompt = [
            personaEnabled ? buildSeniorSupportPersonaBlock() : 'You are a helpful assistant.',
            '',
            'Revise the response to satisfy ALL checklist items below while preserving correct content.',
            'IMPORTANT: Do NOT include headings like "Summary:", "Steps:", "Validation:", "Rollback:", or "Notes:".',
            'Write a natural, conversational answer without meta sections.',
            checklist,
            '',
            'Original response:',
            text,
            '',
            'Revised response (no headings, no boilerplate, answer directly):'
        ].join('\n');

        const revRes = await chatModel.generateContent({
            contents: [{ role: 'user', parts: [{ text: revisePrompt }] }],
            generationConfig: { maxOutputTokens: Math.min(CONFIG.maxResponseTokens, 700), temperature: 0.3, topP: 0.8, topK: 40 }
        });
        const revisedText = revRes.response.text() || text;

        return { text: revisedText, meta: { validated: true, revised: true, verdict } };
    } catch (e) {
        console.warn('Validator failed:', e?.message);
        return { text, meta: { validated: false, revised: false, error: 'validator_failed' } };
    }
}

/**
 * Create a cache key for text
 */
function createCacheKey(text) {
    // Simple hash function for cache key
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
        const char = text.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString();
}

/**
 * Memory management for conversation contexts
 */
function optimizeMemoryUsage() {
    const now = Date.now();
    let cleanedSessions = 0;
    
    // Clean up expired sessions
    for (const [sessionId, context] of conversationContexts.entries()) {
        if (now - context.lastActivity > SESSION_TIMEOUT) {
            conversationContexts.delete(sessionId);
            cleanedSessions++;
        }
    }
    
    // Clean up embedding cache if it's getting too large
    if (embeddingCache.size > EMBEDDING_CACHE_SIZE * 1.2) {
        const entries = Array.from(embeddingCache.entries());
        const sortedEntries = entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
        
        // Keep only the most recent entries
        embeddingCache.clear();
        sortedEntries.slice(-EMBEDDING_CACHE_SIZE).forEach(([key, value]) => {
            embeddingCache.set(key, value);
        });
    }
    
    if (cleanedSessions > 0) {
        console.log(`Memory optimization: cleaned ${cleanedSessions} expired sessions, cache size: ${embeddingCache.size}`);
    }
}

// Run memory optimization every 10 minutes
setInterval(optimizeMemoryUsage, 10 * 60 * 1000);

/**
 * Build MiniSearch index from chunks (BM25-ish keyword retrieval)
 */
async function buildMiniSearchIndex(chunks) {
    try {
        if (!CONFIG.hybrid.enableMiniSearch) {
            miniSearch = null;
            miniSearchReady = false;
            return;
        }

        miniSearch = new MiniSearch({
            fields: ['text'],
            storeFields: ['id', 'text'],
            idField: 'id',
            tokenize: (text, _fieldName) => {
                // simple whitespace tokenizer, lowercase, strip punctuation
                return (text || '')
                    .toLowerCase()
                    .replace(/[^\w\s]/g, ' ')
                    .split(/\s+/)
                    .filter(t => t && t.length > 1);
            }
        });

        const docs = (chunks || []).map(c => ({ id: c.id, text: c.text }));
        await miniSearch.addAllAsync(docs);
        miniSearchReady = true;
    } catch (e) {
        console.warn('Failed to build MiniSearch index:', e?.message);
        miniSearch = null;
        miniSearchReady = false;
    }
}

/**
 * Estimate tokens from text length using heuristic
 */
function estimateTokens(text) {
    const perChar = CONFIG.packing.tokensPerChar || 0.25;
    return Math.ceil((text?.length || 0) * perChar);
}

/**
 * Extract query terms for scoring sentences
 */
function getQueryTerms(query) {
    if (!query) return [];
    return query
        .toLowerCase()
        .replace(/[^\w\s]/g, ' ')
        .split(/\s+/)
        .filter(w => w.length > 2);
}

/**
 * Sentence-level packing under a token budget.
 * Accepts array of { chunk, score, baseScore, ... } and returns similar array with chunk.text reduced.
 */
function packSentencesFromChunks(chunkEntries, maxTokensBudget) {
    if (!Array.isArray(chunkEntries) || chunkEntries.length === 0) return [];

    const sentenceSplit = CONFIG.packing.sentenceSplitRegex || /(?<=[.!?])\s+/;
    const queryTermsSet = new Set(); // optionally filled per-chunk when available

    // Build a flat list of sentence candidates with scoring
    const candidates = [];
    for (const entry of chunkEntries) {
        const text = entry.chunk?.text || '';
        const sentences = text.split(sentenceSplit).map(s => s.trim()).filter(Boolean);
        for (const sentence of sentences) {
            // base from chunk score, plus slight bonus if sentence is longer than a short threshold
            let sScore = (entry.baseScore ?? entry.score ?? 0) + Math.min(sentence.length / 400, 0.05);

            // query term overlap bonus (if we have original query in metadata; not always)
            const terms = entry.metadata?.originalQueryTerms || [];
            if (terms.length > 0) {
                const lower = sentence.toLowerCase();
                let hits = 0;
                for (const t of terms) {
                    if (t.length > 2 && lower.includes(t)) hits++;
                }
                if (hits > 0) {
                    sScore += Math.min(0.02 * hits, 0.08);
                }
            }

            candidates.push({
                chunkId: entry.chunk.id,
                sentence,
                approxTokens: estimateTokens(sentence),
                score: sScore,
                originalEntry: entry
            });
        }
    }

    // Greedy packing by descending score density (score per token)
    candidates.sort((a, b) => (b.score / Math.max(1, b.approxTokens)) - (a.score / Math.max(1, a.approxTokens)));

    let remaining = Math.max(1, maxTokensBudget || 1000);
    const selectedByChunk = new Map();

    for (const c of candidates) {
        if (remaining <= 0) break;
        if (c.approxTokens > remaining) continue;

        const list = selectedByChunk.get(c.chunkId) || [];
        list.push(c);
        selectedByChunk.set(c.chunkId, list);
        remaining -= c.approxTokens;
    }

    // Rebuild packed entries per chunk, preserving metadata
    const packed = [];
    for (const [chunkId, list] of selectedByChunk.entries()) {
        // Keep original entry fields but replace text with joined sentences (original order approximation)
        // Sort sentences by their order in original text when possible
        const entry = list[0].originalEntry;
        const originalText = entry.chunk.text;
        const orderMap = new Map();
        list.forEach(s => orderMap.set(s, originalText.indexOf(s.sentence)));
        list.sort((a, b) => orderMap.get(a) - orderMap.get(b));

        const newText = list.map(s => s.sentence).join(' ');
        const approxCombinedTokens = list.reduce((sum, s) => sum + s.approxTokens, 0);

        packed.push({
            ...entry,
            chunk: {
                ...entry.chunk,
                text: newText
            },
            // keep the same score fields; optionally adjust score slightly by density
            score: entry.score,
            baseScore: entry.baseScore,
            metadata: entry.metadata,
            _packedTokens: approxCombinedTokens
        });
    }

    // If packing ended up empty (e.g., sentences too long), fallback to original entries trimmed to budget
    if (packed.length === 0) {
        let fallbackRemaining = Math.max(1, maxTokensBudget || 1000);
        for (const entry of chunkEntries) {
            const t = entry.chunk.text;
            const approx = estimateTokens(t);
            if (approx <= fallbackRemaining) {
                packed.push(entry);
                fallbackRemaining -= approx;
            } else if (fallbackRemaining > 20) {
                // take a prefix
                const allowedChars = Math.floor(fallbackRemaining / (CONFIG.packing.tokensPerChar || 0.25));
                packed.push({
                    ...entry,
                    chunk: { ...entry.chunk, text: t.slice(0, Math.max(20, allowedChars)) }
                });
                fallbackRemaining = 0;
                break;
            } else {
                break;
            }
        }
    }

    return packed;
}

// Add user contributions to knowledge base
function addUserContribution(question, answer, metadata = {}) {
    if (!knowledgeBase) {
        knowledgeBase = {
            chunks: [],
            embeddings: [],
            metadata: {
                totalChunks: 0,
                lastProcessed: Date.now(),
                version: '1.0'
            }
        };
    }
    
    const contributionText = `${question}\
${answer}`;
    
    // This would be expanded to actually generate embeddings and add to the knowledge base
    // For now, we'll log the contribution
    console.log('User contribution received:', { question, answer, metadata });
    
    return {
        success: true,
        message: 'Thank you for your contribution!'
    };
}

/**
 * Advanced personalization and learning system
 */
function updateUserProfile(sessionId, userAction, context) {
    try {
        const userId = extractUserIdFromSession(sessionId);
        if (!userId) return;
        
        if (!userProfiles.has(userId)) {
            userProfiles.set(userId, {
                preferences: {
                    responseLength: 'medium', // short, medium, long
                    responseStyle: 'helpful', // concise, detailed, helpful, technical
                    topics: new Map(), // topic -> interest score
                    entities: new Map() // entity -> relevance score
                },
                interactions: {
                    totalQueries: 0,
                    successfulInteractions: 0,
                    averageSessionLength: 0,
                    commonQueryTypes: new Map(),
                    timeOfDayPatterns: new Map()
                },
                learning: {
                    responseQualityFeedback: [],
                    topicExpertise: new Map(),
                    conversationPatterns: []
                },
                created: Date.now(),
                lastActivity: Date.now()
            });
        }
        
        const profile = userProfiles.get(userId);
        profile.lastActivity = Date.now();
        
        // Update based on user action
        switch (userAction.type) {
            case 'query':
                updateQueryInteraction(profile, userAction, context);
                break;
            case 'feedback':
                updateFeedbackData(profile, userAction, context);
                break;
            case 'contribution':
                updateContributionData(profile, userAction, context);
                break;
            case 'session_end':
                updateSessionData(profile, userAction, context);
                break;
        }
        
        // Cleanup old data
        cleanupUserProfile(profile);
        
    } catch (error) {
        console.error('Error updating user profile:', error);
    }
}

function updateQueryInteraction(profile, action, context) {
    profile.interactions.totalQueries++;
    
    // Track query type patterns
    const queryType = context.queryAnalysis?.intent || 'unknown';
    const currentCount = profile.interactions.commonQueryTypes.get(queryType) || 0;
    profile.interactions.commonQueryTypes.set(queryType, currentCount + 1);
    
    // Track time of day patterns
    const hour = new Date().getHours();
    const timeSlot = getTimeSlot(hour);
    const timeCount = profile.interactions.timeOfDayPatterns.get(timeSlot) || 0;
    profile.interactions.timeOfDayPatterns.set(timeSlot, timeCount + 1);
    
    // Update topic interests
    if (context.entities) {
        context.entities.forEach(entity => {
            const currentScore = profile.preferences.topics.get(entity.type) || 0;
            profile.preferences.topics.set(entity.type, currentScore + 0.1);
        });
    }
    
    // Infer response style preferences based on query complexity
    const queryLength = action.query.length;
    if (queryLength > 100) {
        // Complex queries suggest preference for detailed responses
        adjustPreference(profile.preferences, 'responseStyle', 'detailed', 0.1);
        adjustPreference(profile.preferences, 'responseLength', 'long', 0.1);
    } else if (queryLength < 30) {
        // Short queries suggest preference for concise responses
        adjustPreference(profile.preferences, 'responseStyle', 'concise', 0.1);
        adjustPreference(profile.preferences, 'responseLength', 'short', 0.1);
    }
}

function updateFeedbackData(profile, action, context) {
    profile.learning.responseQualityFeedback.push({
        responseId: action.responseId,
        rating: action.rating,
        feedback: action.feedback,
        timestamp: Date.now(),
        context: {
            queryType: context.queryType,
            responseLength: context.responseLength,
            confidence: context.confidence
        }
    });
    
    // Learn from positive/negative feedback
    if (action.rating >= 4) {
        profile.interactions.successfulInteractions++;
        // Boost preferences that led to good responses
        if (context.responseStyle) {
            adjustPreference(profile.preferences, 'responseStyle', context.responseStyle, 0.2);
        }
        if (context.responseLength) {
            adjustPreference(profile.preferences, 'responseLength', context.responseLength, 0.2);
        }
    } else if (action.rating <= 2) {
        // Reduce preferences that led to poor responses
        if (context.responseStyle) {
            adjustPreference(profile.preferences, 'responseStyle', context.responseStyle, -0.1);
        }
    }
    
    // Keep only recent feedback (last 100 items)
    if (profile.learning.responseQualityFeedback.length > 100) {
        profile.learning.responseQualityFeedback = profile.learning.responseQualityFeedback.slice(-100);
    }
}

function updateContributionData(profile, action, context) {
    // Track expertise areas based on contributions
    const category = action.category || 'general';
    const currentExpertise = profile.learning.topicExpertise.get(category) || 0;
    profile.learning.topicExpertise.set(category, currentExpertise + 1);
    
    // Contributors tend to prefer more detailed, helpful responses
    adjustPreference(profile.preferences, 'responseStyle', 'helpful', 0.2);
    adjustPreference(profile.preferences, 'responseLength', 'medium', 0.1);
}

function updateSessionData(profile, action, context) {
    const sessionLength = action.sessionDuration || 0;
    
    // Update average session length (exponential moving average)
    const alpha = 0.1; // smoothing factor
    profile.interactions.averageSessionLength = 
        (1 - alpha) * profile.interactions.averageSessionLength + alpha * sessionLength;
}

function adjustPreference(preferences, category, value, adjustment) {
    if (!preferences[category + 'Scores']) {
        preferences[category + 'Scores'] = new Map();
    }
    
    const scores = preferences[category + 'Scores'];
    const currentScore = scores.get(value) || 0;
    scores.set(value, Math.max(0, Math.min(1, currentScore + adjustment)));
    
    // Update the primary preference to the highest scoring option
    let maxScore = 0;
    let bestOption = preferences[category];
    
    for (const [option, score] of scores.entries()) {
        if (score > maxScore) {
            maxScore = score;
            bestOption = option;
        }
    }
    
    preferences[category] = bestOption;
}

function getTimeSlot(hour) {
    if (hour >= 6 && hour < 12) return 'morning';
    if (hour >= 12 && hour < 17) return 'afternoon';
    if (hour >= 17 && hour < 22) return 'evening';
    return 'night';
}

function cleanupUserProfile(profile) {
    const oneWeekAgo = Date.now() - (7 * 24 * 60 * 60 * 1000);
    
    // Remove old feedback
    profile.learning.responseQualityFeedback = profile.learning.responseQualityFeedback
        .filter(feedback => feedback.timestamp > oneWeekAgo);
    
    // Decay topic interests over time
    for (const [topic, score] of profile.preferences.topics.entries()) {
        const decayedScore = score * 0.99; // 1% decay
        if (decayedScore < 0.01) {
            profile.preferences.topics.delete(topic);
        } else {
            profile.preferences.topics.set(topic, decayedScore);
        }
    }
}

function extractUserIdFromSession(sessionId) {
    // In a real system, this would map sessions to authenticated users
    // For now, use sessionId as a proxy for user identification
    return sessionId ? `user_${sessionId.substring(0, 16)}` : null;
}

function getPersonalizedResponseInstructions(sessionId, queryAnalysis) {
    try {
        const userId = extractUserIdFromSession(sessionId);
        if (!userId || !userProfiles.has(userId)) {
            return null;
        }
        
        const profile = userProfiles.get(userId);
        const preferences = profile.preferences;
        
        let instructions = [];
        
        // Response length preference
        switch (preferences.responseLength) {
            case 'short':
                instructions.push('Keep the response concise and to the point (1-2 sentences).');
                break;
            case 'long':
                instructions.push('Provide a comprehensive, detailed response with examples and context.');
                break;
            default:
                instructions.push('Provide a moderately detailed response that covers the key points.');
        }
        
        // Response style preference
        switch (preferences.responseStyle) {
            case 'concise':
                instructions.push('Use clear, direct language without unnecessary elaboration.');
                break;
            case 'detailed':
                instructions.push('Include relevant details, examples, and background information.');
                break;
            case 'technical':
                instructions.push('Use precise, technical language appropriate for an expert audience.');
                break;
            default:
                instructions.push('Use a helpful, friendly tone that balances clarity with completeness.');
        }
        
        // Topic-specific adjustments
        const queryTopics = queryAnalysis.entities
            ?.map(entity => entity.type)
            .filter(type => preferences.topics.has(type)) || [];
        
        if (queryTopics.length > 0) {
            const topicScores = queryTopics.map(topic => preferences.topics.get(topic));
            const avgInterest = topicScores.reduce((sum, score) => sum + score, 0) / topicScores.length;
            
            if (avgInterest > 0.7) {
                instructions.push('The user has shown high interest in this topic, so provide additional depth and related information.');
            } else if (avgInterest < 0.3) {
                instructions.push('Keep the response accessible and avoid overwhelming technical details.');
            }
        }
        
        // Time-based adjustments
        const currentTimeSlot = getTimeSlot(new Date().getHours());
        const timePattern = profile.interactions.timeOfDayPatterns.get(currentTimeSlot) || 0;
        
        if (timePattern > 10 && currentTimeSlot === 'morning') {
            instructions.push('Consider that this is a morning query - the user may want quick, actionable information.');
        }
        
        return instructions.length > 0 ? instructions.join(' ') : null;
        
    } catch (error) {
        console.error('Error generating personalized instructions:', error);
        return null;
    }
}

/**
 * Real-time learning from user interactions
 */
function trackResponseQuality(responseId, sessionId, query, response, confidence, retrievalInfo) {
    const qualityMetrics = {
        responseId,
        sessionId,
        query,
        response,
        confidence,
        retrievalInfo,
        timestamp: Date.now(),
        qualityScore: calculateResponseQuality(query, response, confidence, retrievalInfo),
        userFeedback: null // Will be updated when user provides feedback
    };
    
    responseQualityTracker.set(responseId, qualityMetrics);
    
    // Cleanup old tracking data (keep last 1000 responses)
    if (responseQualityTracker.size > 1000) {
        const oldestEntries = Array.from(responseQualityTracker.entries())
            .sort((a, b) => a[1].timestamp - b[1].timestamp)
            .slice(0, responseQualityTracker.size - 1000);
        
        oldestEntries.forEach(([id]) => responseQualityTracker.delete(id));
    }
}

function calculateResponseQuality(query, response, confidence, retrievalInfo) {
    let qualityScore = 0;
    
    // Base score from confidence
    qualityScore += confidence * 0.4;
    
    // Score from retrieval quality
    if (retrievalInfo && retrievalInfo.bestSimilarity) {
        qualityScore += retrievalInfo.bestSimilarity * 0.3;
    }
    
    // Score from response characteristics
    const responseLength = response.length;
    const queryLength = query.length;
    
    // Penalize very short responses to long queries
    if (queryLength > 50 && responseLength < 100) {
        qualityScore -= 0.1;
    }
    
    // Boost for comprehensive responses to complex queries
    if (queryLength > 100 && responseLength > 200) {
        qualityScore += 0.1;
    }
    
    // Check for helpful phrases
    const helpfulPhrases = [
        'let me help',
        'i can assist',
        'here\'s how',
        'to do this',
        'you can',
        'the solution is'
    ];
    
    const helpfulCount = helpfulPhrases.filter(phrase => 
        response.toLowerCase().includes(phrase)
    ).length;
    
    qualityScore += Math.min(helpfulCount * 0.05, 0.15);
    
    return Math.max(0, Math.min(1, qualityScore));
}

/**
 * ML-based knowledge base optimization
 */
function optimizeKnowledgeBase() {
    if (!knowledgeBase || knowledgeBase.chunks.length === 0) {
        return;
    }
    
    try {
        // Analyze chunk performance
        const chunkPerformance = analyzeChunkPerformance();
        
        // Identify low-quality chunks
        const lowQualityChunks = identifyLowQualityChunks(chunkPerformance);
        
        // Suggest improvements
        const improvements = suggestKnowledgeBaseImprovements(chunkPerformance, lowQualityChunks);
        
        console.log(`Knowledge base optimization: ${improvements.length} suggestions generated`);
        
        return improvements;
        
    } catch (error) {
        console.error('Error optimizing knowledge base:', error);
        return [];
    }
}

function analyzeChunkPerformance() {
    const chunkStats = new Map();
    
    // Initialize stats for all chunks
    knowledgeBase.chunks.forEach(chunk => {
        chunkStats.set(chunk.id, {
            chunk,
            retrievalCount: 0,
            avgSimilarity: 0,
            avgQualityScore: 0,
            lastUsed: null,
            userFeedback: []
        });
    });
    
    // Analyze historical usage
    responseQualityTracker.forEach(metrics => {
        if (metrics.retrievalInfo && metrics.retrievalInfo.chunks) {
            metrics.retrievalInfo.chunks.forEach(chunkResult => {
                const chunkId = chunkResult.chunk.id;
                if (chunkStats.has(chunkId)) {
                    const stats = chunkStats.get(chunkId);
                    stats.retrievalCount++;
                    stats.avgSimilarity = (stats.avgSimilarity + chunkResult.similarity) / 2;
                    stats.avgQualityScore = (stats.avgQualityScore + metrics.qualityScore) / 2;
                    stats.lastUsed = Math.max(stats.lastUsed || 0, metrics.timestamp);
                }
            });
        }
    });
    
    return chunkStats;
}

function identifyLowQualityChunks(chunkPerformance) {
    const lowQualityThreshold = 0.3;
    const lowUsageThreshold = 5;
    const stalenessThreshold = 30 * 24 * 60 * 60 * 1000; // 30 days
    
    const lowQualityChunks = [];
    
    chunkPerformance.forEach((stats, chunkId) => {
        const isLowQuality = stats.avgQualityScore < lowQualityThreshold && stats.retrievalCount > 10;
        const isUnused = stats.retrievalCount < lowUsageThreshold;
        const isStale = stats.lastUsed && (Date.now() - stats.lastUsed) > stalenessThreshold;
        
        if (isLowQuality || isUnused || isStale) {
            lowQualityChunks.push({
                chunkId,
                stats,
                issues: {
                    lowQuality: isLowQuality,
                    unused: isUnused,
                    stale: isStale
                }
            });
        }
    });
    
    return lowQualityChunks;
}

function suggestKnowledgeBaseImprovements(chunkPerformance, lowQualityChunks) {
    const improvements = [];
    
    // Suggest removing or updating low-quality chunks
    lowQualityChunks.forEach(({ chunkId, stats, issues }) => {
        if (issues.unused && stats.retrievalCount === 0) {
            improvements.push({
                type: 'remove',
                chunkId,
                reason: 'Never retrieved - likely irrelevant or duplicate',
                priority: 'low'
            });
        } else if (issues.lowQuality) {
            improvements.push({
                type: 'rewrite',
                chunkId,
                reason: `Low quality score (${stats.avgQualityScore.toFixed(2)}) despite frequent use`,
                priority: 'high',
                suggestion: 'Consider rewriting for clarity and completeness'
            });
        } else if (issues.stale) {
            improvements.push({
                type: 'review',
                chunkId,
                reason: 'Not retrieved recently - may be outdated',
                priority: 'medium',
                suggestion: 'Review and update if necessary'
            });
        }
    });
    
    // Identify content gaps
    const queryTypes = new Map();
    responseQualityTracker.forEach(metrics => {
        if (metrics.confidence < 0.3) { // Low confidence responses indicate gaps
            const queryType = classifyQuery(metrics.query);
            const count = queryTypes.get(queryType) || 0;
            queryTypes.set(queryType, count + 1);
        }
    });
    
    // Suggest new content for frequent low-confidence queries
    queryTypes.forEach((count, queryType) => {
        if (count > 5) {
            improvements.push({
                type: 'add_content',
                queryType,
                reason: `${count} low-confidence responses for ${queryType} queries`,
                priority: 'high',
                suggestion: `Add comprehensive content about ${queryType}`
            });
        }
    });
    
    return improvements.sort((a, b) => {
        const priorityOrder = { high: 3, medium: 2, low: 1 };
        return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
}

function classifyQuery(query) {
    const queryLower = query.toLowerCase();
    
    if (queryLower.includes('how') || queryLower.includes('what') || queryLower.includes('why')) {
        return 'informational';
    } else if (queryLower.includes('when') || queryLower.includes('where')) {
        return 'locational';
    } else if (queryLower.includes('contact') || queryLower.includes('support')) {
        return 'support';
    } else if (queryLower.includes('price') || queryLower.includes('cost')) {
        return 'pricing';
    } else {
        return 'general';
    }
}

// Cleanup function for memory management
function cleanupMemory() {
    const oneHourAgo = Date.now() - (60 * 60 * 1000);
    const oneDayAgo = Date.now() - (24 * 60 * 60 * 1000);
    
    // Cleanup conversation contexts
    for (const [sessionId, context] of conversationContexts.entries()) {
        if (context.lastActivity < oneHourAgo) {
            conversationContexts.delete(sessionId);
        }
    }
    
    // Cleanup conversation memory
    for (const [sessionId, memory] of conversationMemory.entries()) {
        if (memory.lastActivity < oneHourAgo) {
            conversationMemory.delete(sessionId);
        }
    }
    
    // Cleanup old response tracking data
    for (const [responseId, metrics] of responseQualityTracker.entries()) {
        if (metrics.timestamp < oneDayAgo) {
            responseQualityTracker.delete(responseId);
        }
    }
}

// Run cleanup every 30 minutes
setInterval(cleanupMemory, 30 * 60 * 1000);

// Simple runtime log to confirm persona flag at boot (once)
try {
    console.log(`[boot] SUPPORT_PERSONA=${process.env.SUPPORT_PERSONA ?? '(unset)'} -> enabled=${CONFIG.supportPersonaEnabled}`);
} catch (_) {}

// Export for use in contribution endpoint and other modules
module.exports.addUserContribution = addUserContribution;
module.exports.generateEmbedding = generateEmbedding;
module.exports.ensureKnowledgeBase = ensureKnowledgeBase;
module.exports.getOrCreateConversationContext = getOrCreateConversationContext;
module.exports.analyzeQuery = analyzeQuery;
module.exports.performContextAwareRetrieval = performContextAwareRetrieval;
module.exports.performanceMonitor = performanceMonitor;
module.exports.conversationContexts = conversationContexts;
module.exports.embeddingCache = embeddingCache;
module.exports.updateUserProfile = updateUserProfile;
module.exports.trackResponseQuality = trackResponseQuality;
module.exports.optimizeKnowledgeBase = optimizeKnowledgeBase;
module.exports.getPersonalizedResponseInstructions = getPersonalizedResponseInstructions;