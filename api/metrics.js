/**
 * RagZzy Metrics API - Performance and health monitoring endpoint
 */

const { performanceMonitor } = require('./chat.js');

/**
 * Main request handler for metrics endpoint
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
        const { type } = req.query;
        
        switch (type) {
            case 'health':
                const healthStatus = performanceMonitor.getHealthStatus();
                return res.status(200).json({
                    status: healthStatus.status,
                    issues: healthStatus.issues,
                    timestamp: Date.now(),
                    uptime: process.uptime(),
                    memory: process.memoryUsage(),
                    system: {
                        node_version: process.version,
                        platform: process.platform,
                        arch: process.arch
                    }
                });
                
            case 'detailed':
                const detailedMetrics = performanceMonitor.getMetrics();
                return res.status(200).json({
                    ...detailedMetrics,
                    system: {
                        uptime: process.uptime(),
                        memory: process.memoryUsage(),
                        node_version: process.version,
                        platform: process.platform,
                        arch: process.arch
                    }
                });
                
            default:
                // Return basic metrics
                const basicMetrics = performanceMonitor.getMetrics();
                return res.status(200).json({
                    summary: {
                        totalRequests: basicMetrics.requests.total,
                        successRate: basicMetrics.requests.total > 0 ? 
                            (basicMetrics.requests.successful / basicMetrics.requests.total * 100).toFixed(2) + '%' : '0%',
                        averageResponseTime: basicMetrics.requests.averageResponseTime.toFixed(0) + 'ms',
                        activeSessions: basicMetrics.context.activeSessions,
                        aiCalls: basicMetrics.ai.embeddingCalls + basicMetrics.ai.generationCalls
                    },
                    timestamp: Date.now()
                });
        }
        
    } catch (error) {
        console.error('Metrics API error:', error);
        
        res.status(500).json({
            error: 'Failed to retrieve metrics',
            code: 'METRICS_ERROR',
            timestamp: Date.now()
        });
    }
};