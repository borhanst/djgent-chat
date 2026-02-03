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
    const streamToggle = document.getElementById('streamToggle');
    
    if (chatForm) {
        chatForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const stream = streamToggle && streamToggle.checked;
            sendMessage(stream);
        });
    }
    
    // Allow sending with Enter key
    if (messageInput) {
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const stream = streamToggle && streamToggle.checked;
                sendMessage(stream);
            }
        });
    }
}

/**
 * Send a message to the chat API
 */
async function sendMessage(stream = false) {
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
        if (stream) {
            await sendStreamingMessage(message);
        } else {
            await sendRegularMessage(message);
        }
    } catch (error) {
        console.error('Error sending message:', error);
        addMessageToUI('assistant', 'Network error. Please check your connection and try again.');
    } finally {
        setLoading(false);
    }
}

/**
 * Send a regular (non-streaming) message
 */
async function sendRegularMessage(message) {
    const response = await fetch('/chat/api/chat/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        },
        body: JSON.stringify({
            message: message,
            session_id: currentSessionId,
            stream: false
        })
    });
    
    const data = await response.json();
    
    if (response.ok) {
        if (data.session_id) {
            currentSessionId = data.session_id;
        }
        addMessageToUI('assistant', data.answer, data.sources);
        updateVectorCount();
    } else {
        addMessageToUI('assistant', data.error || 'An error occurred. Please try again.');
    }
}

/**
 * Send a streaming message
 */
async function sendStreamingMessage(message) {
    const response = await fetch('/chat/api/chat/stream/', {
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
    
    if (!response.ok) {
        const data = await response.json();
        addMessageToUI('assistant', data.error || 'An error occurred. Please try again.');
        return;
    }
    
    // Create assistant message placeholder (streaming mode - no loading modal)
    const assistantDiv = document.createElement('div');
    assistantDiv.className = `message message-assistant message-new`;
    
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    
    assistantDiv.innerHTML = `
        <div class="message-header">
            <strong>Assistant:</strong>
            <small class="text-muted">${timeStr}</small>
        </div>
        <div class="message-content"></div>
    `;
    
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.appendChild(assistantDiv);
    
    // Remove empty state if present
    const emptyState = chatMessages.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }
    
    const contentDiv = assistantDiv.querySelector('.message-content');
    let fullContent = '';
    
    // Handle streaming response
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        fullContent += chunk;
        contentDiv.innerHTML = escapeHtml(fullContent);
        scrollToBottom();
    }
    
    // Update session ID if changed
    const sessionIdEl = chatMessages.parentElement?.parentElement?.querySelector('.card-body code');
    if (sessionIdEl && currentSessionId) {
        sessionIdEl.textContent = currentSessionId;
    }
    
    updateVectorCount();
    
    // Remove animation class
    setTimeout(() => {
        assistantDiv.classList.remove('message-new');
    }, 300);
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
    
    if (loading) {
        if (sendButton) {
            sendButton.disabled = true;
            sendButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span>';
        }
        if (messageInput) {
            messageInput.disabled = true;
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
