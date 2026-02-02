/**
 * Chat Application JavaScript
 * Handles chat interactions, API calls, and UI updates
 */

// Global variables
let currentSessionId = null;
let isLoading = false;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Set session ID from template
    if (typeof sessionId !== 'undefined') {
        currentSessionId = sessionId;
    }
    
    // Initialize chat form
    initChatForm();
    
    // Scroll to bottom of messages
    scrollToBottom();
});

/**
 * Initialize the chat form
 */
function initChatForm() {
    const chatForm = document.getElementById('chatForm');
    const messageInput = document.getElementById('messageInput');
    
    if (chatForm) {
        chatForm.addEventListener('submit', function(e) {
            e.preventDefault();
            sendMessage();
        });
    }
    
    // Allow sending with Enter key
    if (messageInput) {
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }
}

/**
 * Send a message to the chat API
 */
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message || isLoading) {
        return;
    }
    
    // Clear input
    messageInput.value = '';
    
    // Add user message to UI
    addMessageToUI('user', message);
    
    // Show loading state
    setLoading(true);
    
    try {
        // Send message to API
        const response = await fetch('/chat/api/chat/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify({
                message: message,
                session_id: currentSessionId
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Update session ID if changed
            if (data.session_id) {
                currentSessionId = data.session_id;
            }
            
            // Add assistant message to UI
            addMessageToUI('assistant', data.answer, data.sources);
            
            // Update vector count
            updateVectorCount();
        } else {
            // Show error message
            addMessageToUI('assistant', data.error || 'An error occurred. Please try again.');
        }
    } catch (error) {
        console.error('Error sending message:', error);
        addMessageToUI('assistant', 'Network error. Please check your connection and try again.');
    } finally {
        setLoading(false);
    }
}

/**
 * Add a message to the UI
 */
function addMessageToUI(role, content, sources = null) {
    const chatMessages = document.getElementById('chatMessages');
    
    // Remove empty state if present
    const emptyState = chatMessages.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }
    
    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role} message-new`;
    
    // Get current time
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    
    // Build message HTML
    let sourcesHtml = '';
    if (sources && sources.length > 0) {
        const sourceNames = sources.map(s => s.file_name).join(', ');
        sourcesHtml = `
            <div class="message-sources">
                <small class="text-muted">
                    <i class="bi bi-file-text"></i> Sources: ${sourceNames}
                </small>
            </div>
        `;
    }
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <strong>${role.charAt(0).toUpperCase() + role.slice(1)}:</strong>
            <small class="text-muted">${timeStr}</small>
        </div>
        <div class="message-content">${escapeHtml(content)}</div>
        ${sourcesHtml}
    `;
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    scrollToBottom();
    
    // Remove animation class after animation completes
    setTimeout(() => {
        messageDiv.classList.remove('message-new');
    }, 300);
}

/**
 * Set loading state
 */
function setLoading(loading) {
    isLoading = loading;
    const sendButton = document.getElementById('sendButton');
    const messageInput = document.getElementById('messageInput');
    const loadingModal = document.getElementById('loadingModal');
    
    if (loading) {
        if (sendButton) {
            sendButton.disabled = true;
            sendButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span>';
        }
        if (messageInput) {
            messageInput.disabled = true;
        }
        if (loadingModal) {
            const modal = new bootstrap.Modal(loadingModal);
            modal.show();
        }
    } else {
        if (sendButton) {
            sendButton.disabled = false;
            sendButton.innerHTML = '<i class="bi bi-send"></i>';
        }
        if (messageInput) {
            messageInput.disabled = false;
            messageInput.focus();
        }
        if (loadingModal) {
            const modal = bootstrap.Modal.getInstance(loadingModal);
            if (modal) {
                modal.hide();
            }
        }
    }
}

/**
 * Scroll chat messages to bottom
 */
function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

/**
 * Start a new chat session
 */
async function newChat() {
    try {
        const response = await fetch('/chat/api/new-session/', {
            method: 'GET',
            headers: {
                'X-CSRFToken': getCSRFToken()
            }
        });
        
        const data = await response.json();
        
        if (response.ok && data.session_id) {
            currentSessionId = data.session_id;
            
            // Clear messages
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.innerHTML = `
                <div class="empty-state text-center py-5">
                    <i class="bi bi-chat-square-text display-4 text-muted"></i>
                    <p class="text-muted mt-3">Start a conversation by asking a question!</p>
                </div>
            `;
            
            // Update session info
            updateSessionInfo();
        }
    } catch (error) {
        console.error('Error creating new session:', error);
    }
}

/**
 * Update vector count display
 */
async function updateVectorCount() {
    try {
        const response = await fetch('/chat/api/index-status/', {
            method: 'GET',
            headers: {
                'X-CSRFToken': getCSRFToken()
            }
        });
        
        const data = await response.json();
        
        if (response.ok) {
            const vectorCountEl = document.getElementById('vectorCount');
            if (vectorCountEl) {
                vectorCountEl.textContent = data.vector_count || 0;
            }
        }
    } catch (error) {
        console.error('Error updating vector count:', error);
    }
}

/**
 * Update session info display
 */
function updateSessionInfo() {
    const sessionIdEl = document.querySelector('.card-body code');
    if (sessionIdEl && currentSessionId) {
        sessionIdEl.textContent = currentSessionId;
    }
}

/**
 * Get CSRF token from cookies
 */
function getCSRFToken() {
    const cookies = document.cookie.split(';');
    for (let cookie of cookies) {
        const [name, value] = cookie.trim().split('=');
        if (name === 'csrftoken') {
            return decodeURIComponent(value);
        }
    }
    return '';
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Make functions globally available
window.newChat = newChat;
