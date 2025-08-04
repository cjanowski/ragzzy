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
        this.contributionsEnabled = true;
        
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
            
            // Settings elements
            settingsToggle: document.getElementById('settingsToggle'),
            settingsModal: document.getElementById('settingsModal'),
            closeSettings: document.getElementById('closeSettings'),
            contributionEnabled: document.getElementById('contributionEnabled'),
            saveSettings: document.getElementById('saveSettings'),
            
            // Toast messages
            errorToast: document.getElementById('errorToast'),
            successToast: document.getElementById('successToast'),
            errorMessage: document.getElementById('errorMessage'),
            successMessage: document.getElementById('successMessage'),
            closeError: document.getElementById('closeError'),
            closeSuccess: document.getElementById('closeSuccess'),
            
            // Settings
            contributionToggle: document.getElementById('contributionToggle')
        };
        
        this.init();
        this.initVisualEffects();
        this.initEnhancedUX();
    }
    
    generateSessionId() {
        return 'sess_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }
    
    init() {
        this.setupEventListeners();
        this.loadSettings();
        this.updateWelcomeTime();
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
        
        
        if (this.elements.closeContribution) {
            this.elements.closeContribution.addEventListener('click', () => {
                this.hideContributionModal();
            });
        }
        
        if (this.elements.cancelContribution) {
            this.elements.cancelContribution.addEventListener('click', () => {
                this.hideContributionModal();
            });
        }
        
        if (this.elements.contributionForm) {
            this.elements.contributionForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.submitContribution();
            });
        }
        
        // Character counter for contribution answer
        if (this.elements.contributionAnswer) {
            this.elements.contributionAnswer.addEventListener('input', () => {
                this.updateContributionCharCounter();
            });
        }
        
        // Toast close buttons
        if (this.elements.closeError) {
            this.elements.closeError.addEventListener('click', () => {
                this.hideToast('error');
            });
        }
        
        if (this.elements.closeSuccess) {
            this.elements.closeSuccess.addEventListener('click', () => {
                this.hideToast('success');
            });
        }
        
        // Modal overlay click to close
        if (this.elements.contributionModal) {
            this.elements.contributionModal.addEventListener('click', (e) => {
                if (e.target === this.elements.contributionModal) {
                    this.hideContributionModal();
                }
            });
        }
        
        // Escape key to close modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                if (this.elements.contributionModal && this.elements.contributionModal.style.display === 'flex') {
                    this.hideContributionModal();
                } else if (this.elements.settingsModal && this.elements.settingsModal.style.display === 'flex') {
                    this.hideSettingsModal();
                }
            }
        });
        
        // Contribution toggle
        if (this.elements.contributionToggle) {
            this.elements.contributionToggle.addEventListener('change', (e) => {
                this.contributionsEnabled = e.target.checked;
                this.saveContributionPreference();
                
                // Hide any existing contribution prompts when disabled
                if (!this.contributionsEnabled) {
                    this.hideAllContributionPrompts();
                }
                
                this.showSuccess(this.contributionsEnabled ? 
                    'Contribution prompts enabled' : 
                    'Contribution prompts disabled'
                );
            });
            
            // Load saved preference
            this.loadContributionPreference();
        }

        // Settings events
        if (this.elements.settingsToggle) {
            this.elements.settingsToggle.addEventListener('click', () => {
                this.showSettingsModal();
            });
        }

        if (this.elements.closeSettings) {
            this.elements.closeSettings.addEventListener('click', () => {
                this.hideSettingsModal();
            });
        }

        if (this.elements.saveSettings) {
            this.elements.saveSettings.addEventListener('click', () => {
                this.saveSettings();
            });
        }

        // Settings modal overlay click to close
        if (this.elements.settingsModal) {
            this.elements.settingsModal.addEventListener('click', (e) => {
                if (e.target === this.elements.settingsModal) {
                    this.hideSettingsModal();
                }
            });
        }
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
        if (!this.elements.contributionAnswer || !this.elements.charCount) {
            return;
        }
        
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
        
        // Show typing indicator and progress
        this.showTypingIndicator();
        this.showProgress(true);
        
        try {
            const response = await this.callChatAPI(message);
            this.hideTypingIndicator();
            this.showProgress(false);
            
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
            
            // Play notification sound for bot responses
            this.playNotificationSound();
            
            // Handle contribution prompt if present and enabled in settings
            if (response.contributionPrompt && response.contributionPrompt.show && this.contributionsEnabled) {
                this.showContributionLink(response.contributionPrompt.message);
            }
            
        } catch (error) {
            this.hideTypingIndicator();
            this.showProgress(false);
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
            metadataDiv.innerHTML = `
                <small style=\"color: #6b7280; font-size: 0.75rem;\">
                    Confidence: ${Math.round(metadata.confidence * 100)}%
                    ${metadata.processingTime ? ` • ${metadata.processingTime}ms` : ''}
                </small>
            `;
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
    
    showContributionLink(message) {
        const promptDiv = document.createElement('div');
        promptDiv.className = 'contribution-prompt';
        promptDiv.innerHTML = `
            <h4>💡 Help Improve My Knowledge</h4>
            <p>${message}</p>
            <div class="contribution-actions">
                <button class="btn btn-secondary" onclick="this.parentElement.parentElement.remove()">Not now</button>
                <a href="contributions.html" class="btn btn-primary">Share Knowledge</a>
            </div>
        `;
        
        this.elements.messagesContainer.appendChild(promptDiv);
        this.scrollToBottom();
    }
    
    showContributionModal(question = '') {
        if (!this.elements.contributionModal) {
            console.warn('Contribution modal not found');
            return;
        }
        
        if (this.elements.contributionQuestion) {
            this.elements.contributionQuestion.value = question;
        }
        if (this.elements.contributionAnswer) {
            this.elements.contributionAnswer.value = '';
        }
        if (this.elements.contributionCategory) {
            this.elements.contributionCategory.value = '';
        }
        if (this.elements.contributionConfidence) {
            this.elements.contributionConfidence.value = '4';
        }
        
        this.updateContributionCharCounter();
        
        this.elements.contributionModal.style.display = 'flex';
        
        if (this.elements.contributionAnswer) {
            this.elements.contributionAnswer.focus();
        }
    }
    
    hideContributionModal() {
        if (this.elements.contributionModal) {
            this.elements.contributionModal.style.display = 'none';
        }
    }
    
    async submitContribution() {
        if (!this.elements.contributionForm) {
            this.showError('Contribution form not available');
            return;
        }
        
        const formData = new FormData(this.elements.contributionForm);
        const contribution = {
            question: formData.get('question') ? formData.get('question').trim() : '',
            answer: formData.get('answer') ? formData.get('answer').trim() : '',
            category: formData.get('category') || 'other',
            confidence: parseInt(formData.get('confidence')) || 4,
            source: 'user'
        };
        
        // Validation
        if (!contribution.question || contribution.question.length < 5) {
            this.showError('Please provide a question (at least 5 characters)');
            return;
        }
        
        if (!contribution.answer || contribution.answer.length < 10) {
            this.showError('Please provide an answer (at least 10 characters)');
            return;
        }
        
        if (contribution.answer.length > 2000) {
            this.showError('Answer is too long (maximum 2000 characters)');
            return;
        }
        
        // Show loading state
        if (this.elements.submitContribution) {
            this.elements.submitContribution.classList.add('loading');
            this.elements.submitContribution.disabled = true;
        }
        
        try {
            const response = await fetch('/api/contribute', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(contribution)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.hideContributionModal();
                this.showSuccess(result.message || 'Thank you for your contribution!');
                
                // Show suggested prompts if available
                if (result.suggestedPrompts && result.suggestedPrompts.length > 0) {
                    this.showSuggestedPrompts(result.suggestedPrompts);
                }
            } else {
                this.showError(result.message || 'Failed to add your contribution. Please try again.');
            }
            
        } catch (error) {
            console.error('Contribution error:', error);
            this.showError('Sorry, there was a problem submitting your contribution. Please try again.');
        } finally {
            // Reset loading state
            if (this.elements.submitContribution) {
                this.elements.submitContribution.classList.remove('loading');
                this.elements.submitContribution.disabled = false;
            }
        }
    }
    
    showSuggestedPrompts(prompts) {
        const suggestionDiv = document.createElement('div');
        suggestionDiv.className = 'knowledge-prompt';
        
        let promptsHtml = prompts.map(prompt => 
            `<div class="guided-prompt-item" onclick="ragzzyApp.showContributionModal(\`${prompt.question.replace(/'/g, "\\'")}\`)">                ${prompt.question}            </div>`
        ).join('');
        
        suggestionDiv.innerHTML = `
            <div class=\"prompt-header\">
                <h3>🌟 More Ways to Help</h3>
                <p>Here are some related questions you could help with:</p>
            </div>
            <div class=\"guided-prompts\">${promptsHtml}</div>
            <div class=\"prompt-actions\">
                <button class=\"btn btn-secondary\" onclick=\"this.parentElement.parentElement.remove()\">Maybe later</button>
            </div>
        `;
        
        this.elements.messagesContainer.appendChild(suggestionDiv);
        this.scrollToBottom();
    }

    // Settings Methods
    loadSettings() {
        try {
            const settings = JSON.parse(localStorage.getItem('ragzzy-settings') || '{}');
            
            // Set contribution enabled setting
            const contributionEnabled = settings.contributionEnabled !== false; // Default to true
            if (this.elements.contributionEnabled) {
                this.elements.contributionEnabled.checked = contributionEnabled;
            }
            
            // Store in instance for easy access
            this.settings = {
                contributionEnabled,
                ...settings
            };
        } catch (error) {
            console.error('Error loading settings:', error);
            // Set defaults
            this.settings = {
                contributionEnabled: true
            };
        }
    }

    saveSettings() {
        try {
            const settings = {
                contributionEnabled: this.elements.contributionEnabled?.checked !== false
            };
            
            localStorage.setItem('ragzzy-settings', JSON.stringify(settings));
            this.settings = settings;
            
            this.showSuccess('Settings saved successfully!');
            this.hideSettingsModal();
        } catch (error) {
            console.error('Error saving settings:', error);
            this.showError('Failed to save settings. Please try again.');
        }
    }

    showSettingsModal() {
        if (!this.elements.settingsModal) {
            console.error('Settings modal not found');
            return;
        }

        // Load current settings into form
        if (this.elements.contributionEnabled && this.settings) {
            this.elements.contributionEnabled.checked = this.settings.contributionEnabled !== false;
        }

        this.elements.settingsModal.style.display = 'flex';

        // Focus first interactive element
        const firstCheckbox = this.elements.settingsModal.querySelector('input[type="checkbox"]');
        if (firstCheckbox) {
            firstCheckbox.focus();
        }
    }

    hideSettingsModal() {
        if (this.elements.settingsModal) {
            this.elements.settingsModal.style.display = 'none';
        }
    }
    
    
    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.errorToast.style.display = 'flex';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.hideToast('error');
        }, 5000);
    }
    
    showSuccess(message) {
        this.elements.successMessage.textContent = message;
        this.elements.successToast.style.display = 'flex';
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            this.hideToast('success');
        }, 3000);
    }
    
    hideToast(type) {
        if (type === 'error') {
            this.elements.errorToast.style.display = 'none';
        } else if (type === 'success') {
            this.elements.successToast.style.display = 'none';
        }
    }
    
    // Utility method for external calls
    contributeKnowledge(question, answer) {
        this.showContributionModal(question);
        if (answer) {
            this.elements.contributionAnswer.value = answer;
            this.updateContributionCharCounter();
        }
    }
    
    // Visual Effects System
    initVisualEffects() {
        this.createShootingStars();
        this.createSparkles();
        this.createFloatingParticles();
        
        // Start effects
        this.startShootingStars();
        this.startSparkles();
        this.startFloatingParticles();
    }
    
    createShootingStars() {
        const container = document.getElementById('shootingStars');
        if (!container) return;
        
        // Create 3 shooting stars
        for (let i = 0; i < 3; i++) {
            const star = document.createElement('div');
            star.className = 'shooting-star';
            star.style.animationDelay = `${i * 2}s`;
            container.appendChild(star);
        }
    }
    
    startShootingStars() {
        const container = document.getElementById('shootingStars');
        if (!container) return;
        
        setInterval(() => {
            const stars = container.querySelectorAll('.shooting-star');
            stars.forEach((star, index) => {
                // Random position
                star.style.top = Math.random() * 30 + '%';
                star.style.left = Math.random() * 20 + '%';
                
                // Random animation duration
                star.style.animationDuration = (2 + Math.random() * 2) + 's';
                
                // Restart animation
                star.style.animation = 'none';
                star.offsetHeight; // Trigger reflow
                star.style.animation = null;
            });
        }, 6000); // New shooting star every 6 seconds
    }
    
    createSparkles() {
        const container = document.getElementById('sparkles');
        if (!container) return;
        
        // Create 15 sparkles
        for (let i = 0; i < 15; i++) {
            const sparkle = document.createElement('div');
            sparkle.className = 'sparkle';
            sparkle.style.left = Math.random() * 100 + '%';
            sparkle.style.top = Math.random() * 100 + '%';
            sparkle.style.animationDelay = Math.random() * 2 + 's';
            sparkle.style.animationDuration = (1.5 + Math.random()) + 's';
            container.appendChild(sparkle);
        }
    }
    
    startSparkles() {
        // Sparkles are handled by CSS animations
        const container = document.getElementById('sparkles');
        if (!container) return;
        
        // Occasionally add new sparkles on user interaction
        document.addEventListener('click', (e) => {
            this.createSparkleAt(e.clientX, e.clientY);
        });
        
        // Add sparkles when messages are sent
        const originalAddMessage = this.addMessage.bind(this);
        this.addMessage = function(text, sender, metadata = {}) {
            originalAddMessage(text, sender, metadata);
            
            // Add sparkles around new messages
            setTimeout(() => {
                const messageElements = document.querySelectorAll('.message');
                const latestMessage = messageElements[messageElements.length - 1];
                if (latestMessage) {
                    const rect = latestMessage.getBoundingClientRect();
                    for (let i = 0; i < 3; i++) {
                        this.createSparkleAt(
                            rect.left + Math.random() * rect.width,
                            rect.top + Math.random() * rect.height
                        );
                    }
                }
            }, 100);
        }.bind(this);
    }
    
    createSparkleAt(x, y) {
        const container = document.getElementById('sparkles');
        if (!container) return;
        
        const sparkle = document.createElement('div');
        sparkle.className = 'sparkle';
        sparkle.style.left = x + 'px';
        sparkle.style.top = y + 'px';
        sparkle.style.position = 'fixed';
        sparkle.style.animationDuration = '1s';
        sparkle.style.zIndex = '1000';
        
        container.appendChild(sparkle);
        
        // Remove sparkle after animation
        setTimeout(() => {
            if (sparkle.parentNode) {
                sparkle.parentNode.removeChild(sparkle);
            }
        }, 1000);
    }
    
    createFloatingParticles() {
        const container = document.getElementById('floatingParticles');
        if (!container) return;
        
        // Create 20 floating particles
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 15 + 's';
            particle.style.animationDuration = (10 + Math.random() * 10) + 's';
            container.appendChild(particle);
        }
    }
    
    startFloatingParticles() {
        // Particles are handled by CSS animations
        const container = document.getElementById('floatingParticles');
        if (!container) return;
        
        // Occasionally refresh particles
        setInterval(() => {
            const particles = container.querySelectorAll('.particle');
            particles.forEach(particle => {
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 5 + 's';
            });
        }, 30000); // Refresh every 30 seconds
    }
    
    // Enhanced user experience features
    initEnhancedUX() {
        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K to open contribution modal
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.showContributionModal();
            }
            
            // Ctrl/Cmd + / to focus message input
            if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                e.preventDefault();
                if (this.elements.messageInput) {
                    this.elements.messageInput.focus();
                }
            }
        });
        
        // Add tooltip system
        this.initTooltips();
        
        // Add progress indicators
        this.initProgressIndicators();
        
        // Add sound effects (optional)
        this.initSoundEffects();
    }
    
    initTooltips() {
        const tooltipElements = [
            { id: 'sendButton', text: 'Send message (Enter)' },
            { id: 'statusDot', text: 'Connection status' },
            { id: 'startContributing', text: 'Share your knowledge (Ctrl+K)' }
        ];
        
        tooltipElements.forEach(({ id, text }) => {
            const element = document.getElementById(id);
            if (element) {
                element.setAttribute('title', text);
                element.setAttribute('aria-label', text);
            }
        });
    }
    
    initProgressIndicators() {
        // Add a subtle progress bar for loading states
        const progressBar = document.createElement('div');
        progressBar.id = 'progressBar';
        progressBar.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 0%;
            height: 3px;
            background: linear-gradient(90deg, #ff006e, #8338ec, #3a86ff);
            z-index: 10000;
            transition: width 0.3s ease;
            opacity: 0;
        `;
        document.body.appendChild(progressBar);
        
        this.progressBar = progressBar;
    }
    
    showProgress(show = true) {
        if (!this.progressBar) return;
        
        if (show) {
            this.progressBar.style.opacity = '1';
            this.progressBar.style.width = '70%';
        } else {
            this.progressBar.style.width = '100%';
            setTimeout(() => {
                this.progressBar.style.opacity = '0';
                this.progressBar.style.width = '0%';
            }, 300);
        }
    }
    
    initSoundEffects() {
        // Create audio context for subtle sound effects
        this.audioContext = null;
        
        // Only enable sounds if user has interacted with page
        document.addEventListener('click', () => {
            if (!this.audioContext) {
                try {
                    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
                } catch (e) {
                    console.log('Audio context not supported');
                }
            }
        }, { once: true });
    }
    
    playNotificationSound() {
        if (!this.audioContext) return;
        
        try {
            const oscillator = this.audioContext.createOscillator();
            const gainNode = this.audioContext.createGain();
            
            oscillator.connect(gainNode);
            gainNode.connect(this.audioContext.destination);
            
            oscillator.frequency.setValueAtTime(800, this.audioContext.currentTime);
            oscillator.frequency.setValueAtTime(600, this.audioContext.currentTime + 0.1);
            
            gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.2);
            
            oscillator.start(this.audioContext.currentTime);
            oscillator.stop(this.audioContext.currentTime + 0.2);
        } catch (e) {
            // Silently fail if audio doesn't work
        }
    }
    
    // Contribution Settings Management
    saveContributionPreference() {
        try {
            localStorage.setItem('ragzzy-contributions-enabled', JSON.stringify(this.contributionsEnabled));
        } catch (e) {
            console.warn('Could not save contribution preference');
        }
    }
    
    loadContributionPreference() {
        try {
            const saved = localStorage.getItem('ragzzy-contributions-enabled');
            if (saved !== null) {
                this.contributionsEnabled = JSON.parse(saved);
                if (this.elements.contributionToggle) {
                    this.elements.contributionToggle.checked = this.contributionsEnabled;
                }
            }
        } catch (e) {
            console.warn('Could not load contribution preference');
        }
    }
    
    hideAllContributionPrompts() {
        // Remove contribution prompts from messages
        const contributionPrompts = document.querySelectorAll('.contribution-prompt');
        contributionPrompts.forEach(prompt => {
            prompt.remove();
        });
        
        // Hide contribution modal if open
        this.hideContributionModal();
    }
}

// Initialize the app when DOM is loaded
let ragzzyApp;

document.addEventListener('DOMContentLoaded', () => {
    ragzzyApp = new RagzzyChatApp();
});

// Make app globally available for onclick handlers
window.ragzzyApp = ragzzyApp;