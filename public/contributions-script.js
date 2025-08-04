/**
 * RagZzy Contributions Interface - Standalone Knowledge Management
 */

class RagzzyContributionsApp {
    constructor() {
        this.contributions = [];
        this.stats = {
            total: 0,
            yours: 0,
            helpful: 0
        };
        
        // DOM elements
        this.elements = {
            // Stats
            totalContributions: document.getElementById('totalContributions'),
            yourContributions: document.getElementById('yourContributions'),
            helpfulVotes: document.getElementById('helpfulVotes'),
            
            // Quick prompts
            quickPrompts: document.getElementById('quickPrompts'),
            
            // Detailed form
            detailedForm: document.getElementById('detailedContributionForm'),
            detailedQuestion: document.getElementById('detailedQuestion'),
            detailedAnswer: document.getElementById('detailedAnswer'),
            detailedCategory: document.getElementById('detailedCategory'),
            detailedTags: document.getElementById('detailedTags'),
            detailedConfidence: document.getElementById('detailedConfidence'),
            detailedCharCount: document.getElementById('detailedCharCount'),
            clearForm: document.getElementById('clearForm'),
            submitDetailed: document.getElementById('submitDetailed'),
            
            // Recent contributions
            recentContributions: document.getElementById('recentContributions'),
            
            // Status
            statusDot: document.getElementById('statusDot'),
            statusText: document.getElementById('statusText'),
            
            // Toast messages
            errorToast: document.getElementById('errorToast'),
            successToast: document.getElementById('successToast'),
            errorMessage: document.getElementById('errorMessage'),
            successMessage: document.getElementById('successMessage'),
            closeError: document.getElementById('closeError'),
            closeSuccess: document.getElementById('closeSuccess')
        };
        
        // Password gate
        this.passwordKey = 'ragzzy_contrib_password';
        this.password = this.getStoredPassword();

        if (this.hasAccess()) {
            this.init();
            this.initVisualEffects();
        } else {
            this.showPasswordGate();
        }
    }
    
    init() {
        this.setupEventListeners();
        this.loadStats();
        this.loadQuickPrompts();
        this.loadRecentContributions();
        this.checkOnlineStatus();
        this.updateCharCounter();
    }
    
    setupEventListeners() {
        // Detailed form submission
        if (this.elements.detailedForm) {
            this.elements.detailedForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.submitDetailedContribution();
            });
        }
        
        // Character counter for detailed answer
        if (this.elements.detailedAnswer) {
            this.elements.detailedAnswer.addEventListener('input', () => {
                this.updateCharCounter();
            });
        }
        
        // Clear form button
        if (this.elements.clearForm) {
            this.elements.clearForm.addEventListener('click', () => {
                this.clearDetailedForm();
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
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter to submit form
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                if (document.activeElement === this.elements.detailedAnswer) {
                    e.preventDefault();
                    this.submitDetailedContribution();
                }
            }
            
            // Escape to clear form
            if (e.key === 'Escape') {
                if (document.activeElement === this.elements.detailedAnswer ||
                    document.activeElement === this.elements.detailedQuestion) {
                    this.clearDetailedForm();
                }
            }
        });
    }
    
    getStoredPassword() {
        try {
            return localStorage.getItem(this.passwordKey) || sessionStorage.getItem(this.passwordKey) || '';
        } catch (_) {
            return '';
        }
    }

    hasAccess() {
        // If server requires a password, we store and send it via header.
        // Client-side gate simply checks that some password exists.
        return !!this.password;
    }

    showPasswordGate() {
        // Create a simple inline modal if HTML doesn't include one
        let gate = document.getElementById('contribPasswordGate');
        if (!gate) {
            gate = document.createElement('div');
            gate.id = 'contribPasswordGate';
            gate.style.position = 'fixed';
            gate.style.inset = '0';
            gate.style.background = 'rgba(0,0,0,0.8)';
            gate.style.display = 'flex';
            gate.style.alignItems = 'center';
            gate.style.justifyContent = 'center';
            gate.style.zIndex = '9999';
            gate.innerHTML = `
                <div style="background:#111827; padding:24px; border-radius:12px; width:90%; max-width:420px; color:#fff; border:1px solid #374151;">
                    <h3 style="margin:0 0 12px 0; font-size:18px;">Enter Contributions Password</h3>
                    <p style="margin:0 0 16px 0; color:#9CA3AF; font-size:14px;">Access to the contributions page is restricted.</p>
                    <div style="display:flex; flex-direction:column; gap:12px;">
                        <input id="contribPasswordInput" type="password" placeholder="Password" style="padding:10px 12px; border-radius:8px; border:1px solid #374151; background:#0B1220; color:#fff; outline:none;">
                        <label style="display:flex; align-items:center; gap:8px; font-size:13px; color:#9CA3AF;">
                            <input id="contribRemember" type="checkbox"> Remember on this device
                        </label>
                        <div style="display:flex; gap:8px; justify-content:flex-end;">
                            <button id="contribCancel" style="padding:8px 12px; background:#374151; color:#fff; border:none; border-radius:8px; cursor:pointer;">Cancel</button>
                            <button id="contribSubmit" style="padding:8px 12px; background:#6366F1; color:#fff; border:none; border-radius:8px; cursor:pointer;">Unlock</button>
                        </div>
                        <div id="contribError" style="color:#ef4444; font-size:13px; display:none;">Invalid password. Try again.</div>
                    </div>
                </div>
            `;
            document.body.appendChild(gate);
        } else {
            gate.style.display = 'flex';
        }

        const input = document.getElementById('contribPasswordInput');
        const remember = document.getElementById('contribRemember');
        const btn = document.getElementById('contribSubmit');
        const cancel = document.getElementById('contribCancel');
        const err = document.getElementById('contribError');

        const submit = () => {
            const pw = input.value.trim();
            if (!pw) {
                err.textContent = 'Password required';
                err.style.display = 'block';
                return;
            }
            // Store and init
            try {
                if (remember.checked) {
                    localStorage.setItem(this.passwordKey, pw);
                } else {
                    sessionStorage.setItem(this.passwordKey, pw);
                }
                this.password = pw;
            } catch (_) {}
            gate.style.display = 'none';
            this.init();
            this.initVisualEffects();
        };

        btn.addEventListener('click', submit);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') submit();
        });
        cancel.addEventListener('click', () => {
            // navigate away to home
            window.location.href = 'index.html';
        });
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
        if (this.elements.statusDot && this.elements.statusText) {
            this.elements.statusDot.className = `status-dot ${isOnline ? 'online' : 'offline'}`;
            this.elements.statusText.textContent = isOnline ? 'Online' : 'Offline';
        }
    }
    
    updateCharCounter() {
        if (!this.elements.detailedAnswer || !this.elements.detailedCharCount) {
            return;
        }
        
        const count = this.elements.detailedAnswer.value.length;
        this.elements.detailedCharCount.textContent = count;
        
        if (count > 4500) {
            this.elements.detailedCharCount.style.color = '#ef4444';
        } else if (count > 4000) {
            this.elements.detailedCharCount.style.color = '#f59e0b';
        } else {
            this.elements.detailedCharCount.style.color = 'rgba(255, 255, 255, 0.7)';
        }
    }
    
    async loadStats() {
        try {
            // Show loading state
            if (this.elements.totalContributions) {
                this.elements.totalContributions.textContent = '...';
                this.elements.yourContributions.textContent = '...';
                this.elements.helpfulVotes.textContent = '...';
            }
            
            const response = await fetch('/api/contribute/stats');
            if (response.ok) {
                const stats = await response.json();
                this.stats = stats;
                this.updateStatsDisplay();
            } else {
                // Show default stats if API call fails
                this.updateStatsDisplay();
            }
        } catch (error) {
            console.error('Error loading stats:', error);
            this.updateStatsDisplay();
        }
    }
    
    updateStatsDisplay() {
        if (this.elements.totalContributions) {
            this.elements.totalContributions.textContent = this.stats.total || 0;
            this.elements.yourContributions.textContent = this.stats.yours || 0;
            this.elements.helpfulVotes.textContent = this.stats.helpful || 0;
        }
    }
    
    async loadQuickPrompts() {
        if (!this.elements.quickPrompts) return;
        
        try {
            // Show loading state
            this.elements.quickPrompts.innerHTML = `
                <div class="loading-placeholder">
                    <span class="loading-spinner"></span>
                    Loading suggested questions...
                </div>
            `;
            
            const response = await fetch('/api/contribute/prompts');
            if (response.ok) {
                const prompts = await response.json();
                this.renderQuickPrompts(prompts.prompts || []);
            } else {
                this.renderDefaultQuickPrompts();
            }
        } catch (error) {
            console.error('Error loading quick prompts:', error);
            this.renderDefaultQuickPrompts();
        }
    }
    
    renderQuickPrompts(prompts) {
        if (!this.elements.quickPrompts) return;
        
        if (prompts.length === 0) {
            this.elements.quickPrompts.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">💡</div>
                    <h4>No Quick Prompts Available</h4>
                    <p>Use the detailed form below to contribute your knowledge.</p>
                </div>
            `;
            return;
        }
        
        const promptsHtml = prompts.map(prompt => `
            <div class="quick-prompt-card" onclick="contributionsApp.fillQuickPrompt('${prompt.question.replace(/'/g, "\\'")}', '${prompt.category}')">
                <h4>${prompt.question}</h4>
                <p>${prompt.description || 'Help other users by answering this commonly asked question.'}</p>
                <span class="category-tag">${this.formatCategory(prompt.category)}</span>
            </div>
        `).join('');
        
        this.elements.quickPrompts.innerHTML = promptsHtml;
    }
    
    renderDefaultQuickPrompts() {
        const defaultPrompts = [
            {
                question: "What are your business hours?",
                category: "business-info",
                description: "Help users know when they can expect support or service."
            },
            {
                question: "How can customers contact support?",
                category: "support",
                description: "Provide contact information and preferred communication methods."
            },
            {
                question: "What are your main products or services?",
                category: "products",
                description: "Give an overview of what your organization offers."
            },
            {
                question: "What is your return/refund policy?",
                category: "policies",
                description: "Explain the terms and conditions for returns and refunds."
            },
            {
                question: "How do I reset my password?",
                category: "support",
                description: "Provide step-by-step instructions for password recovery."
            },
            {
                question: "What payment methods do you accept?",
                category: "business-info",
                description: "List accepted payment options and any related details."
            }
        ];
        
        this.renderQuickPrompts(defaultPrompts);
    }
    
    fillQuickPrompt(question, category) {
        if (this.elements.detailedQuestion) {
            this.elements.detailedQuestion.value = question;
        }
        if (this.elements.detailedCategory) {
            this.elements.detailedCategory.value = category;
        }
        if (this.elements.detailedAnswer) {
            this.elements.detailedAnswer.focus();
        }
        
        // Scroll to form
        const formSection = document.querySelector('.detailed-contribute-section');
        if (formSection) {
            formSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
    
    async loadRecentContributions() {
        if (!this.elements.recentContributions) return;
        
        try {
            // Show loading state
            this.elements.recentContributions.innerHTML = `
                <div class="loading-placeholder">
                    <span class="loading-spinner"></span>
                    Loading recent contributions...
                </div>
            `;
            
            const response = await fetch('/api/contribute/recent');
            if (response.ok) {
                const data = await response.json();
                this.renderRecentContributions(data.contributions || []);
            } else {
                this.renderEmptyContributions();
            }
        } catch (error) {
            console.error('Error loading recent contributions:', error);
            this.renderEmptyContributions();
        }
    }
    
    renderRecentContributions(contributions) {
        if (!this.elements.recentContributions) return;
        
        if (contributions.length === 0) {
            this.renderEmptyContributions();
            return;
        }
        
        const contributionsHtml = contributions.map(contribution => `
            <div class="contribution-item">
                <div class="contribution-header">
                    <div class="contribution-question">${contribution.question}</div>
                    <div class="contribution-meta">
                        <span>Confidence: ${contribution.confidence}/5</span>
                        <span>${this.formatDate(contribution.timestamp)}</span>
                    </div>
                </div>
                <div class="contribution-answer">${this.truncateText(contribution.answer, 200)}</div>
                <div class="contribution-footer">
                    <span class="contribution-category">${this.formatCategory(contribution.category)}</span>
                    <div class="contribution-actions">
                        <button onclick="contributionsApp.voteHelpful('${contribution.id}')">👍 Helpful</button>
                        <button onclick="contributionsApp.reportContribution('${contribution.id}')">⚠️ Report</button>
                    </div>
                </div>
            </div>
        `).join('');
        
        this.elements.recentContributions.innerHTML = contributionsHtml;
    }
    
    renderEmptyContributions() {
        if (!this.elements.recentContributions) return;
        
        this.elements.recentContributions.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-icon">📚</div>
                <h4>No Contributions Yet</h4>
                <p>Be the first to share your knowledge and help build our community resource.</p>
            </div>
        `;
    }
    
    async submitDetailedContribution() {
        if (!this.elements.detailedForm) return;
        
        const formData = new FormData(this.elements.detailedForm);
        const contribution = {
            question: formData.get('question')?.trim() || '',
            answer: formData.get('answer')?.trim() || '',
            category: formData.get('category') || 'other',
            tags: formData.get('tags')?.trim() || '',
            confidence: parseInt(formData.get('confidence')) || 4,
            source: 'contributions-page'
        };
        
        // Validation
        if (!contribution.question || contribution.question.length < 5) {
            this.showError('Please provide a question (at least 5 characters)');
            this.elements.detailedQuestion?.focus();
            return;
        }
        
        if (!contribution.answer || contribution.answer.length < 20) {
            this.showError('Please provide a detailed answer (at least 20 characters)');
            this.elements.detailedAnswer?.focus();
            return;
        }
        
        if (contribution.answer.length > 5000) {
            this.showError('Answer is too long (maximum 5000 characters)');
            this.elements.detailedAnswer?.focus();
            return;
        }
        
        if (!contribution.category) {
            this.showError('Please select a category');
            this.elements.detailedCategory?.focus();
            return;
        }
        
        // Show loading state
        if (this.elements.submitDetailed) {
            this.elements.submitDetailed.classList.add('loading');
            this.elements.submitDetailed.disabled = true;
        }
        
        try {
            const doPost = async () => {
                const headers = {
                    'Content-Type': 'application/json',
                };
                if (this.password) {
                    headers['X-Contrib-Auth'] = this.password;
                }
                const resp = await fetch('/api/contribute', {
                    method: 'POST',
                    headers,
                    body: JSON.stringify(contribution)
                });
                return resp;
            };

            let response = await doPost();

            // Handle unauthorized: prompt for password and retry once
            if (response.status === 401) {
                this.showPasswordGate();
                // Wait a tick for user interaction is out of scope; instead, show error and return.
                this.showError('Unauthorized. Please enter the contributions password.');
                return;
            }

            const result = await response.json();
            
            if (result.success) {
                this.showSuccess(result.message || 'Thank you for your contribution!');
                this.clearDetailedForm();
                
                // Refresh data
                this.loadStats();
                this.loadRecentContributions();
                this.loadQuickPrompts();
                
                // Scroll to top to show success
                window.scrollTo({ top: 0, behavior: 'smooth' });
            } else {
                this.showError(result.message || 'Failed to add your contribution. Please try again.');
            }
            
        } catch (error) {
            console.error('Contribution error:', error);
            this.showError('Sorry, there was a problem submitting your contribution. Please try again.');
        } finally {
            // Reset loading state
            if (this.elements.submitDetailed) {
                this.elements.submitDetailed.classList.remove('loading');
                this.elements.submitDetailed.disabled = false;
            }
        }
    }
    
    clearDetailedForm() {
        if (this.elements.detailedForm) {
            this.elements.detailedForm.reset();
            this.updateCharCounter();
            
            // Focus first field
            if (this.elements.detailedQuestion) {
                this.elements.detailedQuestion.focus();
            }
        }
    }
    
    async voteHelpful(contributionId) {
        try {
            const headers = { 'Content-Type': 'application/json' };
            if (this.password) headers['X-Contrib-Auth'] = this.password;

            const response = await fetch(`/api/contribute/${contributionId}/vote`, {
                method: 'POST',
                headers,
                body: JSON.stringify({ vote: 'helpful' })
            });
            
            if (response.ok) {
                this.showSuccess('Thank you for your feedback!');
                this.loadStats(); // Refresh stats
            } else {
                this.showError('Unable to record your vote. Please try again.');
            }
        } catch (error) {
            console.error('Vote error:', error);
            this.showError('Unable to record your vote. Please try again.');
        }
    }
    
    async reportContribution(contributionId) {
        if (confirm('Are you sure you want to report this contribution? This will flag it for review.')) {
            try {
                const headers = { 'Content-Type': 'application/json' };
                if (this.password) headers['X-Contrib-Auth'] = this.password;

                const response = await fetch(`/api/contribute/${contributionId}/report`, {
                    method: 'POST',
                    headers
                });
                
                if (response.ok) {
                    this.showSuccess('Thank you for reporting. We will review this contribution.');
                } else {
                    this.showError('Unable to report this contribution. Please try again.');
                }
            } catch (error) {
                console.error('Report error:', error);
                this.showError('Unable to report this contribution. Please try again.');
            }
        }
    }
    
    // Utility methods
    formatCategory(category) {
        const categoryMap = {
            'business-info': 'Business Info',
            'products': 'Products',
            'pricing': 'Pricing',
            'policies': 'Policies',
            'support': 'Support',
            'technical': 'Technical',
            'other': 'Other'
        };
        return categoryMap[category] || category;
    }
    
    formatDate(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diffTime = Math.abs(now - date);
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
        
        if (diffDays === 1) {
            return 'Today';
        } else if (diffDays === 2) {
            return 'Yesterday';
        } else if (diffDays <= 7) {
            return `${diffDays - 1} days ago`;
        } else {
            return date.toLocaleDateString();
        }
    }
    
    truncateText(text, maxLength) {
        if (text.length <= maxLength) {
            return text;
        }
        return text.substring(0, maxLength) + '...';
    }
    
    showError(message) {
        if (this.elements.errorMessage && this.elements.errorToast) {
            this.elements.errorMessage.textContent = message;
            this.elements.errorToast.style.display = 'flex';
            
            // Auto-hide after 5 seconds
            setTimeout(() => {
                this.hideToast('error');
            }, 5000);
        }
    }
    
    showSuccess(message) {
        if (this.elements.successMessage && this.elements.successToast) {
            this.elements.successMessage.textContent = message;
            this.elements.successToast.style.display = 'flex';
            
            // Auto-hide after 3 seconds
            setTimeout(() => {
                this.hideToast('success');
            }, 3000);
        }
    }
    
    hideToast(type) {
        if (type === 'error' && this.elements.errorToast) {
            this.elements.errorToast.style.display = 'none';
        } else if (type === 'success' && this.elements.successToast) {
            this.elements.successToast.style.display = 'none';
        }
    }
    
    // Visual Effects (reuse from main app)
    initVisualEffects() {
        this.createShootingStars();
        this.createSparkles();
        this.createFloatingParticles();
        
        this.startShootingStars();
        this.startSparkles();
        this.startFloatingParticles();
    }
    
    createShootingStars() {
        const container = document.getElementById('shootingStars');
        if (!container) return;
        
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
            stars.forEach((star) => {
                star.style.top = Math.random() * 30 + '%';
                star.style.left = Math.random() * 20 + '%';
                star.style.animationDuration = (2 + Math.random() * 2) + 's';
                
                star.style.animation = 'none';
                star.offsetHeight; // Trigger reflow
                star.style.animation = null;
            });
        }, 6000);
    }
    
    createSparkles() {
        const container = document.getElementById('sparkles');
        if (!container) return;
        
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
        document.addEventListener('click', (e) => {
            this.createSparkleAt(e.clientX, e.clientY);
        });
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
        
        setTimeout(() => {
            if (sparkle.parentNode) {
                sparkle.parentNode.removeChild(sparkle);
            }
        }, 1000);
    }
    
    createFloatingParticles() {
        const container = document.getElementById('floatingParticles');
        if (!container) return;
        
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
        const container = document.getElementById('floatingParticles');
        if (!container) return;
        
        setInterval(() => {
            const particles = container.querySelectorAll('.particle');
            particles.forEach(particle => {
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 5 + 's';
            });
        }, 30000);
    }
}

// Initialize the app when DOM is loaded
let contributionsApp;

document.addEventListener('DOMContentLoaded', () => {
    contributionsApp = new RagzzyContributionsApp();
});

// Make app globally available
window.contributionsApp = contributionsApp;