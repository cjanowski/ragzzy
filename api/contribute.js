/**
 * RagZzy Contribution API - Handles user knowledge contributions
 * Processes user-submitted knowledge and adds it to the knowledge base
 */

const fs = require('fs');
const path = require('path');

// Configuration
const CONFIG = {
    minQuestionLength: 5,
    minAnswerLength: 10,
    maxAnswerLength: parseInt(process.env.MAX_CONTRIBUTION_LENGTH) || 2000,
    allowedCategories: ['business-info', 'products', 'pricing', 'policies', 'support', 'other']
};

/**
 * Main contribution handler
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
        const contribution = validateContribution(req.body);
        
        if (!contribution.valid) {
            return res.status(400).json({
                success: false,
                message: contribution.error,
                code: 'VALIDATION_ERROR',
                requestId
            });
        }
        
        // Process the contribution
        const result = await processContribution(contribution.data, requestId);
        const processingTime = Date.now() - startTime;
        
        // Return success response
        res.status(200).json({
            ...result,
            timestamp: Date.now(),
            processingTime,
            requestId
        });
        
    } catch (error) {
        console.error(`[${requestId}] Contribution API error:`, error);
        
        const processingTime = Date.now() - startTime;
        
        // Handle specific error types
        if (error.message?.includes('file system')) {
            return res.status(500).json({
                success: false,
                message: 'Unable to save contribution at this time. Please try again.',
                code: 'STORAGE_ERROR',
                requestId,
                processingTime
            });
        }
        
        if (error.message?.includes('API key') || error.message?.includes('embedding')) {
            return res.status(500).json({
                success: false,
                message: 'Service temporarily unavailable. Your contribution has been saved and will be processed soon.',
                code: 'SERVICE_ERROR',
                requestId,
                processingTime
            });
        }
        
        // Generic error response
        res.status(500).json({
            success: false,
            message: 'Something went wrong while processing your contribution. Please try again.',
            code: 'INTERNAL_ERROR',
            requestId,
            processingTime
        });
    }
};

/**
 * Validate contribution data
 */
function validateContribution(data) {
    if (!data || typeof data !== 'object') {
        return {
            valid: false,
            error: 'Invalid contribution data'
        };
    }
    
    const { question, answer, category, confidence, source } = data;
    
    // Validate question
    if (!question || typeof question !== 'string') {
        return {
            valid: false,
            error: 'Question is required'
        };
    }
    
    if (question.trim().length < CONFIG.minQuestionLength) {
        return {
            valid: false,
            error: `Question must be at least ${CONFIG.minQuestionLength} characters long`
        };
    }
    
    if (question.length > 500) {
        return {
            valid: false,
            error: 'Question is too long (maximum 500 characters)'
        };
    }
    
    // Validate answer
    if (!answer || typeof answer !== 'string') {
        return {
            valid: false,
            error: 'Answer is required'
        };
    }
    
    if (answer.trim().length < CONFIG.minAnswerLength) {
        return {
            valid: false,
            error: `Answer must be at least ${CONFIG.minAnswerLength} characters long`
        };
    }
    
    if (answer.length > CONFIG.maxAnswerLength) {
        return {
            valid: false,
            error: `Answer is too long (maximum ${CONFIG.maxAnswerLength} characters)`
        };
    }
    
    // Validate category (optional)
    if (category && !CONFIG.allowedCategories.includes(category)) {
        return {
            valid: false,
            error: 'Invalid category'
        };
    }
    
    // Validate confidence (optional)
    if (confidence !== undefined) {
        const conf = parseInt(confidence);
        if (isNaN(conf) || conf < 1 || conf > 5) {
            return {
                valid: false,
                error: 'Confidence must be a number between 1 and 5'
            };
        }
    }
    
    // Check for spam/inappropriate content
    if (isInappropriateContent(question + ' ' + answer)) {
        return {
            valid: false,
            error: 'Content appears to be inappropriate or spam'
        };
    }
    
    return {
        valid: true,
        data: {
            question: question.trim(),
            answer: answer.trim(),
            category: category || 'other',
            confidence: parseInt(confidence) || 4,
            source: source || 'user'
        }
    };
}

/**
 * Process the validated contribution
 */
async function processContribution(contribution, requestId) {
    try {
        // Save contribution to knowledge base file
        await saveToKnowledgeBase(contribution, requestId);
        
        // Generate follow-up suggestions
        const suggestedPrompts = generateFollowUpPrompts(contribution);
        const relatedTopics = findRelatedTopics(contribution.category);
        
        // Log the contribution for analytics
        logContribution(contribution, requestId);
        
        return {
            success: true,
            message: getThankYouMessage(contribution.category),
            suggestedPrompts: suggestedPrompts,
            relatedTopics: relatedTopics,
            contribution: {
                category: contribution.category,
                confidence: contribution.confidence,
                wordCount: contribution.answer.split(' ').length
            }
        };
        
    } catch (error) {
        console.error(`[${requestId}] Error processing contribution:`, error);
        throw error;
    }
}

/**
 * Save contribution to knowledge base file
 */
async function saveToKnowledgeBase(contribution, requestId) {
    try {
        const knowledgeBasePath = path.join(process.cwd(), 'knowledge_base.txt');
        
        // Format the contribution
        const timestamp = new Date().toISOString();
        const formattedContribution = `\
\
${contribution.question}\
${contribution.answer}\
\
# User contribution - ${timestamp} - Category: ${contribution.category} - Confidence: ${contribution.confidence}/5`;
        
        // Append to knowledge base file
        await fs.promises.appendFile(knowledgeBasePath, formattedContribution, 'utf8');
        
        console.log(`[${requestId}] Contribution saved to knowledge base`);
        
    } catch (error) {
        console.error(`[${requestId}] Error saving to knowledge base:`, error);
        throw new Error('file system error');
    }
}

/**
 * Generate follow-up prompts based on the contribution
 */
function generateFollowUpPrompts(contribution) {
    const allPrompts = {
        'business-info': [
            {
                id: 'location',
                question: 'Where are you located?',
                category: 'business-info'
            },
            {
                id: 'team-size',
                question: 'How many people work at your company?',
                category: 'business-info'
            },
            {
                id: 'founded',
                question: 'When was your company founded?',
                category: 'business-info'
            }
        ],
        'products': [
            {
                id: 'features',
                question: 'What are the key features of your main product?',
                category: 'products'
            },
            {
                id: 'target-audience',
                question: 'Who is your target audience?',
                category: 'products'
            },
            {
                id: 'competitors',
                question: 'How do you differ from competitors?',
                category: 'products'
            }
        ],
        'pricing': [
            {
                id: 'payment-methods',
                question: 'What payment methods do you accept?',
                category: 'pricing'
            },
            {
                id: 'discounts',
                question: 'Do you offer any discounts or promotions?',
                category: 'pricing'
            },
            {
                id: 'billing',
                question: 'How does billing work?',
                category: 'pricing'
            }
        ],
        'policies': [
            {
                id: 'privacy',
                question: 'What is your privacy policy?',
                category: 'policies'
            },
            {
                id: 'terms',
                question: 'What are your terms of service?',
                category: 'policies'
            },
            {
                id: 'warranty',
                question: 'What warranty do you provide?',
                category: 'policies'
            }
        ],
        'support': [
            {
                id: 'response-time',
                question: 'How quickly do you respond to support requests?',
                category: 'support'
            },
            {
                id: 'troubleshooting',
                question: 'How can users troubleshoot common issues?',
                category: 'support'
            },
            {
                id: 'documentation',
                question: 'Where can users find documentation?',
                category: 'support'
            }
        ],
        'other': [
            {
                id: 'general-1',
                question: 'What are your business hours?',
                category: 'business-info'
            },
            {
                id: 'general-2',
                question: 'How can customers contact you?',
                category: 'business-info'
            },
            {
                id: 'general-3',
                question: 'What makes your service unique?',
                category: 'products'
            }
        ]
    };
    
    const categoryPrompts = allPrompts[contribution.category] || allPrompts['other'];
    
    // Return 2-3 random prompts from the category
    const shuffled = categoryPrompts.sort(() => 0.5 - Math.random());
    return shuffled.slice(0, Math.min(3, shuffled.length));
}

/**
 * Find related topics based on category
 */
function findRelatedTopics(category) {
    const relatedTopics = {
        'business-info': ['Contact Information', 'Company Background', 'Office Locations', 'Team'],
        'products': ['Features', 'Use Cases', 'Integrations', 'Roadmap'],
        'pricing': ['Plans', 'Payment', 'Billing', 'Discounts'],
        'policies': ['Terms of Service', 'Privacy Policy', 'Refunds', 'Warranties'],
        'support': ['Help Center', 'Tutorials', 'Troubleshooting', 'Community'],
        'other': ['General Information', 'Getting Started', 'Best Practices']
    };
    
    return relatedTopics[category] || relatedTopics['other'];
}

/**
 * Get a personalized thank you message
 */
function getThankYouMessage(category) {
    const messages = {
        'business-info': 'Thank you for helping others learn more about the company!',
        'products': 'Great! Your product knowledge will help many users.',
        'pricing': 'Thanks for clarifying pricing information - very helpful!',
        'policies': 'Excellent! Policy information is crucial for users.',
        'support': 'Thank you for sharing support knowledge!',
        'other': 'Thank you for your valuable contribution!'
    };
    
    return messages[category] || messages['other'];
}

/**
 * Log contribution for analytics
 */
function logContribution(contribution, requestId) {
    const logEntry = {
        timestamp: new Date().toISOString(),
        requestId,
        category: contribution.category,
        confidence: contribution.confidence,
        questionLength: contribution.question.length,
        answerLength: contribution.answer.length,
        source: contribution.source
    };
    
    // In a production system, this would go to a proper logging service
    console.log('Contribution logged:', JSON.stringify(logEntry));
}

/**
 * Check for inappropriate content
 */
function isInappropriateContent(text) {
    const inappropriatePatterns = [
        /\\b(spam|viagra|casino|porn|xxx)\\b/i,
        /\\b(buy now|click here|free money|get rich quick)\\b/i,
        /\\b(f[u*]ck|sh[i*]t|d[a*]mn|b[i*]tch)\\b/i,
        /\\b(\\w)\\1{5,}/g, // Excessive repeated characters
        /^.{0,10}$/g, // Very short responses that are likely spam
    ];
    
    // Check for excessive caps (more than 70% uppercase)
    const uppercaseRatio = (text.match(/[A-Z]/g) || []).length / text.length;
    if (uppercaseRatio > 0.7 && text.length > 20) {
        return true;
    }
    
    // Check for excessive punctuation
    const punctuationRatio = (text.match(/[!?]{3,}/g) || []).length;
    if (punctuationRatio > 2) {
        return true;
    }
    
    return inappropriatePatterns.some(pattern => pattern.test(text));
}

/**
 * Generate a unique request ID
 */
function generateRequestId() {
    return 'contrib_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
}

/**
 * Health check endpoint for contribution service
 */
module.exports.healthCheck = () => {
    return {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        config: {
            minQuestionLength: CONFIG.minQuestionLength,
            minAnswerLength: CONFIG.minAnswerLength,
            maxAnswerLength: CONFIG.maxAnswerLength,
            allowedCategories: CONFIG.allowedCategories
        }
    };
};