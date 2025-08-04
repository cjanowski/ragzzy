/**
 * RagZzy Chat API - Main chat endpoint with RAG functionality
 * Handles user queries, retrieval-augmented generation, and knowledge contribution prompts
 */

const { GoogleGenerativeAI } = require('@google/generative-ai');
const fs = require('fs');
const path = require('path');

// Initialize Gemini AI
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const chatModel = genAI.getGenerativeModel({ model: 'gemini-2.0-flash-exp' });
const embeddingModel = genAI.getGenerativeModel({ model: 'text-embedding-004' });

// In-memory knowledge base cache
let knowledgeBase = null;
let lastProcessed = 0;

// Configuration
const CONFIG = {
    confidenceThreshold: parseFloat(process.env.CONFIDENCE_THRESHOLD) || 0.3,
    maxRetrievalChunks: 5,
    chunkSize: 500,
    chunkOverlap: 50,
    maxResponseTokens: 1000
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
        const { message, sessionId } = req.body;
        
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
        
        // Initialize or update knowledge base
        await ensureKnowledgeBase();
        
        // Process chat request
        const response = await processChatRequest(message, sessionId, requestId);
        const processingTime = Date.now() - startTime;
        
        // Return response
        res.status(200).json({
            ...response,
            timestamp: Date.now(),
            processingTime,
            requestId
        });
        
    } catch (error) {
        console.error(`[${requestId}] Chat API error:`, error);
        
        const processingTime = Date.now() - startTime;
        
        // Handle specific error types
        if (error.message?.includes('API key')) {
            return res.status(500).json({
                error: 'Service temporarily unavailable',
                code: 'SERVICE_ERROR',
                requestId,
                processingTime
            });
        }
        
        if (error.message?.includes('quota') || error.message?.includes('rate limit')) {
            return res.status(429).json({
                error: 'Too many requests. Please try again in a moment.',
                code: 'RATE_LIMITED',
                requestId,
                processingTime
            });
        }
        
        // Generic error response
        res.status(500).json({
            error: 'Something went wrong. Please try again.',
            code: 'INTERNAL_ERROR',
            requestId,
            processingTime
        });
    }
};

/**
 * Process the chat request using RAG
 */
async function processChatRequest(message, sessionId, requestId) {
    try {
        // Generate embedding for user query
        const queryEmbedding = await generateEmbedding(message);
        
        // Retrieve relevant knowledge chunks
        const retrievedChunks = searchSimilarChunks(queryEmbedding, CONFIG.maxRetrievalChunks);
        
        // Calculate confidence based on best similarity score
        const confidence = retrievedChunks.length > 0 ? retrievedChunks[0].similarity : 0;
        
        // Generate response using retrieved context
        const response = await generateResponse(message, retrievedChunks);
        
        // Determine if we should prompt for contribution
        const shouldPromptContribution = confidence < CONFIG.confidenceThreshold && 
                                       message.length > 10 && 
                                       !isSpamOrInappropriate(message);
        
        const result = {
            response: response,
            confidence: confidence,
            sources: retrievedChunks.map(chunk => chunk.metadata?.source || 'knowledge_base').filter(Boolean)
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
        console.error(`[${requestId}] RAG processing error:`, error);
        throw error;
    }
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
            
            console.log(`Knowledge base loaded: ${knowledgeBase.chunks.length} chunks`);
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
 * Generate embedding for text
 */
async function generateEmbedding(text) {
    try {
        const result = await embeddingModel.embedContent(text);
        return result.embedding.values;
    } catch (error) {
        console.error('Error generating embedding:', error);
        throw error;
    }
}

/**
 * Search for similar chunks using cosine similarity
 */
function searchSimilarChunks(queryEmbedding, topK = 5) {
    if (!knowledgeBase || knowledgeBase.chunks.length === 0) {
        return [];
    }
    
    const similarities = knowledgeBase.chunks.map(chunk => ({
        chunk,
        similarity: cosineSimilarity(queryEmbedding, chunk.embedding),
        metadata: chunk.metadata
    }));
    
    return similarities
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, topK)
        .filter(result => result.similarity > 0.1); // Filter out very low similarity
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
 * Generate response using retrieved context
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

// Export for use in contribution endpoint
module.exports.addUserContribution = addUserContribution;
module.exports.generateEmbedding = generateEmbedding;