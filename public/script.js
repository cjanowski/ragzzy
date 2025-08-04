/**
 * RagZzy Chat Interface - Frontend Logic
 * Handles chat interactions, knowledge contributions, and UI management
 */

class RagzzyChatApp {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.messages = [];
        this.isTyping = false;
        this.guidedPrompts = [];
        
        // DOM elements
        this.elements = {
            messagesContainer: document.getElementById('messagesContainer'),
            messageForm: document.getElementById('messageForm'),
            messageInput: document.getElementById('messageInput'),
            sendButton: document.getElementById('sendButton'),
            typingIndicator: document.getElementById('typingIndicator'),
            charCounter: document.getElementById('charCounter'),
            statusDot: document.getElementById('statusDot'),
            statusText: document.getElementById('statusText'),
            
            // Knowledge contribution elements
            knowledgePrompt: document.getElementById('knowledgePrompt'),
            contributionModal: document.getElementById('contributionModal'),
            contributionForm: document.getElementById('contributionForm'),
            contributionQuestion: document.getElementById('contributionQuestion'),
            contributionAnswer: document.getElementById('contributionAnswer'),
            contributionCategory: document.getElementById('contributionCategory'),
            contributionConfidence: document.getElementById('contributionConfidence'),
            charCount: document.getElementById('charCount'),
            
            // Buttons
            startContributing: document.getElementById('startContributing'),
            skipPrompts: document.getElementById('skipPrompts'),
            closeContribution: document.getElementById('closeContribution'),
            cancelContribution: document.getElementById('cancelContribution'),
            submitContribution: document.getElementById('submitContribution'),
            
            // Toast messages
            errorToast: document.getElementById('errorToast'),
            successToast: document.getElementById('successToast'),
            errorMessage: document.getElementById('errorMessage'),
            successMessage: document.getElementById('successMessage'),
            closeError: document.getElementById('closeError'),
            closeSuccess: document.getElementById('closeSuccess')
        };
        
        this.init();
    }
    
    generateSessionId() {
        return 'sess_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }
    
    init() {
        this.setupEventListeners();
        this.updateWelcomeTime();
        this.loadGuidedPrompts();
        this.updateCharCounter();
        this.checkOnlineStatus();
        
        // Set initial focus
        this.elements.messageInput.focus();
    }
    
    setupEventListeners() {
        // Message form submission
        this.elements.messageForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        // Message input handling
        this.elements.messageInput.addEventListener('input', () => {
            this.updateCharCounter();
            this.toggleSendButton();
            this.autoResize();
        });
        
        this.elements.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Knowledge contribution events
        this.elements.startContributing.addEventListener('click', () => {
            this.showContributionModal();
        });
        
        this.elements.skipPrompts.addEventListener('click', () => {
            this.hideKnowledgePrompt();
        });
        
        this.elements.closeContribution.addEventListener('click', () => {
            this.hideContributionModal();
        });
        
        this.elements.cancelContribution.addEventListener('click', () => {
            this.hideContributionModal();
        });
        
        this.elements.contributionForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.submitContribution();
        });
        
        // Character counter for contribution answer
        this.elements.contributionAnswer.addEventListener('input', () => {
            this.updateContributionCharCounter();
        });
        
        // Toast close buttons
        this.elements.closeError.addEventListener('click', () => {
            this.hideToast('error');
        });
        
        this.elements.closeSuccess.addEventListener('click', () => {
            this.hideToast('success');
        });
        
        // Modal overlay click to close
        this.elements.contributionModal.addEventListener('click', (e) => {
            if (e.target === this.elements.contributionModal) {
                this.hideContributionModal();
            }
        });
        
        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.elements.contributionModal.style.display === 'flex') {
                this.hideContributionModal();
            }
        });
    }
    
    updateWelcomeTime() {
        const welcomeTime = document.getElementById('welcomeTime');
        if (welcomeTime) {
            welcomeTime.textContent = new Date().toLocaleTimeString();
        }
    }
    
    updateCharCounter() {
        const count = this.elements.messageInput.value.length;
        this.elements.charCounter.textContent = `${count}/1000`;
        
        if (count > 900) {
            this.elements.charCounter.style.color = '#ef4444';
        } else if (count > 800) {
            this.elements.charCounter.style.color = '#f59e0b';
        } else {
            this.elements.charCounter.style.color = '#6b7280';
        }
    }
    
    updateContributionCharCounter() {
        const count = this.elements.contributionAnswer.value.length;
        this.elements.charCount.textContent = count;
        
        if (count > 1800) {
            this.elements.charCount.style.color = '#ef4444';
        } else if (count > 1600) {
            this.elements.charCount.style.color = '#f59e0b';
        } else {
            this.elements.charCount.style.color = '#6b7280';
        }
    }
    
    toggleSendButton() {
        const hasContent = this.elements.messageInput.value.trim().length > 0;
        this.elements.sendButton.disabled = !hasContent || this.isTyping;
    }
    
    autoResize() {
        const textarea = this.elements.messageInput;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }
    
    async checkOnlineStatus() {
        try {
            const response = await fetch('/api/health', { method: 'GET' });
            const isOnline = response.ok;
            this.updateStatus(isOnline);
        } catch (error) {
            this.updateStatus(false);
        }
    }
    
    updateStatus(isOnline) {
        this.elements.statusDot.className = `status-dot ${isOnline ? 'online' : 'offline'}`;
        this.elements.statusText.textContent = isOnline ? 'Online' : 'Offline';
    }
    
    async sendMessage() {
        const message = this.elements.messageInput.value.trim();
        if (!message || this.isTyping) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input and update UI
        this.elements.messageInput.value = '';
        this.autoResize();
        this.updateCharCounter();
        this.toggleSendButton();
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            const response = await this.callChatAPI(message);
            this.hideTypingIndicator();
            
            if (response.error) {
                this.showError(response.error);
                return;
            }
            
            // Add bot response
            this.addMessage(response.response, 'bot', {
                confidence: response.confidence,
                sources: response.sources,
                processingTime: response.processingTime
            });
            
            // Handle contribution prompt if present
            if (response.contributionPrompt && response.contributionPrompt.show) {
                this.showContributionPrompt(response.contributionPrompt.message, message);
            }
            
        } catch (error) {
            this.hideTypingIndicator();
            this.showError('Sorry, I\'m having trouble connecting. Please try again.');
            console.error('Chat API error:', error);
        }
    }
    
    async callChatAPI(message) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                sessionId: this.sessionId,
                timestamp: Date.now()
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    }
    
    addMessage(text, sender, metadata = {}) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        messageDiv.setAttribute('role', 'article');
        
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.innerHTML = `<div class="avatar ${sender}-avatar">${sender === 'user' ? '👤' : '🤖'}</div>`;
        
        const content = document.createElement('div');
        content.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        messageText.innerHTML = this.formatMessage(text);
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = new Date().toLocaleTimeString();
        
        // Add metadata for bot messages
        if (sender === 'bot' && metadata.confidence !== undefined) {
            const metadataDiv = document.createElement('div');
            metadataDiv.className = 'message-metadata';
            metadataDiv.innerHTML = `\n                <small style=\"color: #6b7280; font-size: 0.75rem;\">\n                    Confidence: ${Math.round(metadata.confidence * 100)}%\n                    ${metadata.processingTime ? ` • ${metadata.processingTime}ms` : ''}\n                </small>\n            `;
            messageText.appendChild(metadataDiv);
        }
        
        content.appendChild(messageText);
        content.appendChild(messageTime);
        
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
        
        this.elements.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Store message
        this.messages.push({
            text,
            sender,
            timestamp: Date.now(),
            metadata
        });
    }
    
    formatMessage(text) {
        // Basic formatting - convert line breaks and handle basic markdown
        return text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>');
    }
    
    showTypingIndicator() {
        this.isTyping = true;
        this.elements.typingIndicator.style.display = 'flex';
        this.toggleSendButton();
        this.scrollToBottom();
    }
    
    hideTypingIndicator() {
        this.isTyping = false;
        this.elements.typingIndicator.style.display = 'none';
        this.toggleSendButton();
    }
    
    scrollToBottom() {
        requestAnimationFrame(() => {
            this.elements.messagesContainer.scrollTop = this.elements.messagesContainer.scrollHeight;
        });
    }
    
    showContributionPrompt(message, originalQuestion) {
        const promptDiv = document.createElement('div');
        promptDiv.className = 'contribution-prompt';
        promptDiv.innerHTML = `\n            <h4>💡 Help Improve My Knowledge</h4>\n            <p>${message}</p>\n            <div class=\"contribution-actions\">\n                <button class=\"btn btn-secondary\" onclick=\"this.parentElement.parentElement.remove()\">Not now</button>\n                <button class=\"btn btn-primary\" onclick=\"ragzzyApp.showContributionModal('${originalQuestion.replace(/'/g, \"\\'\")}')\">Help Out</button>\n            </div>\n        `;
        
        this.elements.messagesContainer.appendChild(promptDiv);
        this.scrollToBottom();
    }
    
    showContributionModal(question = '') {
        this.elements.contributionQuestion.value = question;
        this.elements.contributionAnswer.value = '';
        this.elements.contributionCategory.value = '';
        this.elements.contributionConfidence.value = '4';
        this.updateContributionCharCounter();
        
        this.elements.contributionModal.style.display = 'flex';
        this.elements.contributionAnswer.focus();
    }
    
    hideContributionModal() {
        this.elements.contributionModal.style.display = 'none';
    }
    
    async submitContribution() {
        const formData = new FormData(this.elements.contributionForm);
        const contribution = {
            question: formData.get('question').trim(),
            answer: formData.get('answer').trim(),
            category: formData.get('category') || 'other',
            confidence: parseInt(formData.get('confidence')),
            source: 'user'
        };
        
        // Validation\n        if (!contribution.question || contribution.question.length < 5) {\n            this.showError('Please provide a question (at least 5 characters)');\n            return;\n        }\n        \n        if (!contribution.answer || contribution.answer.length < 10) {\n            this.showError('Please provide an answer (at least 10 characters)');\n            return;\n        }\n        \n        if (contribution.answer.length > 2000) {\n            this.showError('Answer is too long (maximum 2000 characters)');\n            return;\n        }\n        \n        // Show loading state\n        this.elements.submitContribution.classList.add('loading');\n        this.elements.submitContribution.disabled = true;\n        \n        try {\n            const response = await fetch('/api/contribute', {\n                method: 'POST',\n                headers: {\n                    'Content-Type': 'application/json',\n                },\n                body: JSON.stringify(contribution)\n            });\n            \n            const result = await response.json();\n            \n            if (result.success) {\n                this.hideContributionModal();\n                this.showSuccess(result.message || 'Thank you for your contribution!');\n                \n                // Show suggested prompts if available\n                if (result.suggestedPrompts && result.suggestedPrompts.length > 0) {\n                    this.showSuggestedPrompts(result.suggestedPrompts);\n                }\n            } else {\n                this.showError(result.message || 'Failed to add your contribution. Please try again.');\n            }\n            \n        } catch (error) {\n            console.error('Contribution error:', error);\n            this.showError('Sorry, there was a problem submitting your contribution. Please try again.');\n        } finally {\n            // Reset loading state\n            this.elements.submitContribution.classList.remove('loading');\n            this.elements.submitContribution.disabled = false;\n        }\n    }\n    \n    showSuggestedPrompts(prompts) {\n        const suggestionDiv = document.createElement('div');\n        suggestionDiv.className = 'knowledge-prompt';\n        \n        let promptsHtml = prompts.map(prompt => \n            `<div class=\"guided-prompt-item\" onclick=\"ragzzyApp.showContributionModal('${prompt.question.replace(/'/g, \"\\'\")}')\">                ${prompt.question}            </div>`\n        ).join('');\n        \n        suggestionDiv.innerHTML = `\n            <div class=\"prompt-header\">\n                <h3>🌟 More Ways to Help</h3>\n                <p>Here are some related questions you could help with:</p>\n            </div>\n            <div class=\"guided-prompts\">${promptsHtml}</div>\n            <div class=\"prompt-actions\">\n                <button class=\"btn btn-secondary\" onclick=\"this.parentElement.parentElement.remove()\">Maybe later</button>\n            </div>\n        `;\n        \n        this.elements.messagesContainer.appendChild(suggestionDiv);\n        this.scrollToBottom();\n    }\n    \n    loadGuidedPrompts() {\n        // Default guided prompts - in a real app, these might come from the API\n        this.guidedPrompts = [\n            {\n                id: 'business-hours',\n                question: 'What are your business hours?',\n                category: 'business-info',\n                required: true\n            },\n            {\n                id: 'contact-info',\n                question: 'How can customers contact support?',\n                category: 'business-info',\n                required: true\n            },\n            {\n                id: 'main-products',\n                question: 'What are your main products or services?',\n                category: 'products',\n                required: true\n            },\n            {\n                id: 'pricing-info',\n                question: 'What is your pricing structure?',\n                category: 'pricing',\n                required: false\n            },\n            {\n                id: 'return-policy',\n                question: 'What is your return/refund policy?',\n                category: 'policies',\n                required: false\n            }\n        ];\n        \n        // Show knowledge prompt after a delay\n        setTimeout(() => {\n            this.showKnowledgePrompt();\n        }, 5000);\n    }\n    \n    showKnowledgePrompt() {\n        const guidedPromptsHtml = this.guidedPrompts.map(prompt => \n            `<div class=\"guided-prompt-item ${prompt.required ? 'required' : ''}\" onclick=\"ragzzyApp.showContributionModal('${prompt.question.replace(/'/g, \"\\'\")}')\">                ${prompt.question}            </div>`\n        ).join('');\n        \n        document.getElementById('guidedPrompts').innerHTML = guidedPromptsHtml;\n        this.elements.knowledgePrompt.style.display = 'block';\n        this.scrollToBottom();\n    }\n    \n    hideKnowledgePrompt() {\n        this.elements.knowledgePrompt.style.display = 'none';\n    }\n    \n    showError(message) {\n        this.elements.errorMessage.textContent = message;\n        this.elements.errorToast.style.display = 'flex';\n        \n        // Auto-hide after 5 seconds\n        setTimeout(() => {\n            this.hideToast('error');\n        }, 5000);\n    }\n    \n    showSuccess(message) {\n        this.elements.successMessage.textContent = message;\n        this.elements.successToast.style.display = 'flex';\n        \n        // Auto-hide after 3 seconds\n        setTimeout(() => {\n            this.hideToast('success');\n        }, 3000);\n    }\n    \n    hideToast(type) {\n        if (type === 'error') {\n            this.elements.errorToast.style.display = 'none';\n        } else if (type === 'success') {\n            this.elements.successToast.style.display = 'none';\n        }\n    }\n    \n    // Utility method for external calls\n    contributeKnowledge(question, answer) {\n        this.showContributionModal(question);\n        if (answer) {\n            this.elements.contributionAnswer.value = answer;\n            this.updateContributionCharCounter();\n        }\n    }\n}\n\n// Initialize the app when DOM is loaded\nlet ragzzyApp;\n\ndocument.addEventListener('DOMContentLoaded', () => {\n    ragzzyApp = new RagzzyChatApp();\n});\n\n// Make app globally available for onclick handlers\nwindow.ragzzyApp = ragzzyApp;"