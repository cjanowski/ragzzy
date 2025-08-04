# RagZzy - Intelligent Customer Support Chatbot

A RAG-powered customer support chatbot that learns from user interactions and contributions.

## Features

- 🤖 **AI-Powered Responses**: Uses Google Gemini for intelligent, context-aware responses
- 📚 **Dynamic Knowledge Base**: Learns from user contributions in real-time
- 🎯 **Guided Knowledge Collection**: Prompts users with specific questions to build comprehensive knowledge
- 🔍 **Smart Retrieval**: Uses semantic search to find relevant information
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
- ⚡ **Fast & Reliable**: Deployed on Vercel with optimized performance

## How It Works

### Core RAG Process

1. **User Query**: User asks a question through the chat interface
2. **Semantic Search**: System finds relevant information in the knowledge base
3. **Context Retrieval**: Most relevant chunks are selected for context
4. **Response Generation**: Gemini generates a response using retrieved context
5. **Knowledge Gaps**: When confidence is low, users are prompted to contribute

### Knowledge Building

- **Interactive Prompts**: Users see guided questions to help structure contributions
- **Real-time Updates**: New knowledge is immediately available for future queries  
- **Quality Control**: Contributions are validated and formatted automatically
- **Smart Suggestions**: System suggests related topics based on user input

## Project Structure

```
ragzzy/
├── api/
│   ├── chat.js          # Main chat endpoint
│   └── contribute.js    # Knowledge contribution endpoint
├── public/
│   ├── index.html       # Chat interface
│   ├── style.css        # Styling
│   └── script.js        # Frontend logic
├── tests/               # Test files
├── knowledge_base.txt   # Initial knowledge base
├── requirements.md      # Detailed requirements
├── design.md           # Technical design document
├── tasks.md            # Project task management
└── vercel.json         # Deployment configuration
```

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Required. Your Google Gemini API key
- `NODE_ENV`: Environment (development/production)
- `LOG_LEVEL`: Logging verbosity (info/debug/error)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for responses (default: 0.3)
- `MAX_CONTRIBUTION_LENGTH`: Maximum contribution length (default: 2000)
- `MIN_CONTRIBUTION_LENGTH`: Minimum contribution length (default: 10)

### Knowledge Base Format

The `knowledge_base.txt` file uses a simple Q&A format:

```
Question or topic
Answer or explanation for the topic.

Another question
Another answer with more details.
```

Each Q&A pair should be separated by a blank line.

## API Endpoints

### POST /api/chat

Chat with the bot.

**Request:**
```json
{
  "message": "What are your business hours?",
  "sessionId": "optional-session-id"
}
```

**Response:**
```json
{
  "response": "Our business hours are...",
  "confidence": 0.95,
  "sources": ["knowledge_base.txt"],
  "timestamp": 1234567890,
  "processingTime": 1250,
  "contributionPrompt": {
    "show": false
  }
}
```

### POST /api/contribute

Contribute knowledge to the system.

**Request:**
```json
{
  "question": "What is your return policy?",
  "answer": "We offer 30-day returns...",
  "category": "policies",
  "confidence": 5
}
```

**Response:**
```json
{
  "success": true,
  "message": "Thank you! Your knowledge has been added.",
  "suggestedPrompts": [...],
  "relatedTopics": [...]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For questions or issues:
- Create an issue in this repository
- Email: support@ragzzy.com
- Check the knowledge base in the chat interface# ragzzy
