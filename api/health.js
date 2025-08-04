/**
 * Health Check API - Simple endpoint to check service status
 */

module.exports = async (req, res) => {
    // Set CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    
    if (req.method === 'OPTIONS') {
        return res.status(200).end();
    }
    
    if (req.method !== 'GET') {
        return res.status(405).json({ 
            error: 'Method not allowed',
            code: 'METHOD_NOT_ALLOWED'
        });
    }
    
    try {
        const healthStatus = {
            status: 'healthy',
            timestamp: new Date().toISOString(),
            service: 'RagZzy Chat API',
            version: '1.0.0',
            uptime: process.uptime(),
            environment: process.env.NODE_ENV || 'development',
            features: {
                chat: true,
                knowledgeContribution: true,
                ragRetrieval: true
            }
        };
        
        // Check if Gemini API key is configured
        if (!process.env.GEMINI_API_KEY) {
            healthStatus.status = 'degraded';
            healthStatus.warnings = ['Gemini API key not configured'];
        }
        
        const statusCode = healthStatus.status === 'healthy' ? 200 : 503;
        res.status(statusCode).json(healthStatus);
        
    } catch (error) {
        console.error('Health check error:', error);
        
        res.status(503).json({
            status: 'unhealthy',
            timestamp: new Date().toISOString(),
            service: 'RagZzy Chat API',
            error: 'Service experiencing issues'
        });
    }
};