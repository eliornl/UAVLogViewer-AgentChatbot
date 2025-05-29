<template>
  <div class="agentic-chat-container">
    <!-- Header Section -->
    <header class="chat-header">
      <div class="header-content">
        <h3><i class="fas fa-robot" aria-hidden="true"></i> UAV Log Viewer Agent Chat Bot</h3>
      </div>
      <div class="session-info" v-if="sessionId">
        <small class="text-muted">Session: {{ sessionId }}</small>
      </div>
    </header>

    <!-- Chat Interface -->
    <main class="chat-interface" :class="{ 'no-file-loaded': !hasUploadedFile }">
      <!-- Chat Messages Area -->
      <div class="chat-messages" ref="chatMessages" aria-relevant="additions" aria-live="polite" v-if="hasUploadedFile">
        <div v-for="message in messages" :key="message.id" class="message-container" :class="message.role">
          <div class="message-content">
            <div v-if="message.role === 'system'" class="system-message">
              <i class="fas fa-info-circle" aria-hidden="true"></i>
              <span v-html="formatMessageContent(message.content)"></span>
            </div>
            <div v-else-if="message.role === 'user'" class="user-message">
              <div class="message-header">
                <span class="message-time">{{ formatTime(message.timestamp) }}</span>
              </div>
              <div class="message-body">
                <span v-html="formatMessageContent(message.content)"></span>
              </div>
            </div>
            <div v-else-if="message.role === 'assistant'" class="assistant-message">
              <div class="message-header">
                <span class="message-time">{{ formatTime(message.timestamp) }}</span>
              </div>
              <div class="message-body">
                <span v-if="message.isStreaming" class="typing-indicator">
                  Processing request...
                </span>
                <span v-else v-html="formatMessageContent(message.content)"></span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Fallback for when no file is loaded yet by parent -->
      <div v-else class="initial-prompt-area-fallback">
        <h5><i class="fas fa-spinner fa-spin" aria-hidden="true"></i> Waiting for Data...</h5>
        <p class="text-muted">
          Please upload a log file or select sample data via the main interface to begin analysis.
        </p>
      </div>

      <!-- Chat Input -->
      <div class="chat-input-section" role="form">
        <div class="custom-input-group">
          <input type="text"
                 class="custom-form-control"
                 v-model="currentMessage"
                 @keyup.enter="sendMessage"
                 :disabled="!hasUploadedFile || isTyping"
                 placeholder="What do you want to know?"
                 aria-label="Chat message input">
          <button class="custom-send-button"
                  @click="sendMessage"
                  :disabled="!currentMessage.trim() || !hasUploadedFile || isTyping"
                  aria-label="Send message">
            Send
          </button>
        </div>

        <!-- Quick Questions in chat input section with improved styling -->
        <div class="quick-questions-container mt-2 position-relative">
          <button class="custom-quick-questions-button"
                  @click="toggleQuickQuestions"
                  :disabled="isTyping || !hasUploadedFile">
            <i class="fas fa-lightbulb" aria-hidden="true"></i> Quick Questions
          </button>

          <div v-if="showQuickQuestions" class="quick-questions-list">
            <div class="quick-questions-header">
              <span>Suggested Questions</span>
              <button class="close-quick-questions" @click="toggleQuickQuestions">
                <i class="fas fa-times" aria-hidden="true"></i>
              </button>
            </div>
            <div v-for="question in quickQuestions" :key="question"
                 class="quick-question-item"
                 @click="askQuickQuestion(question)">
              <i class="fas fa-question-circle" aria-hidden="true"></i> {{ question }}
            </div>
          </div>
        </div>
      </div>
    </main>

    <!-- Session Actions -->
    <footer class="session-actions" v-if="hasUploadedFile">
      <!-- Message count removed as requested -->
      <div class="action-buttons">
        <!-- Left section - Download Chat -->
        <div class="download-section">
          <div class="custom-download-group">
            <button class="custom-download-btn"
                    @click="downloadChat"
                    :disabled="messages.length === 0">
              Download Chat
            </button>
            <select class="custom-format-select"
                    v-model="downloadFormat"
                    aria-label="Select download format">
              <option value="text">(.txt)</option>
              <option value="json">(.json)</option>
              <option value="csv">(.csv)</option>
            </select>
          </div>
        </div>

        <!-- Middle section - Placeholder -->
        <div class="middle-placeholder"></div>

        <!-- Right section - placeholder for future buttons -->
        <div class="right-actions"></div>
      </div>
    </footer>
  </div>
</template>

<script>
import axios from 'axios'
import globalConfig from '../../config' // Corrected import

const MESSAGE_ROLES = {
    USER: 'user',
    ASSISTANT: 'assistant',
    SYSTEM: 'system'
}

const VALID_FILE_EXTENSIONS = ['.bin', '.log', '.tlog', '.ulg', '.ulog']

export default {
    name: 'AgenticChat',
    props: {
    /**
     * The session ID provided by the parent component after successful file upload/processing.
     * @type {String}
     */
        initialSessionId: {
            type: String,
            default: null
        },
        /**
     * The name of the file associated with the session, for display purposes.
     * @type {String}
     */
        initialFileName: {
            type: String,
            default: 'your log file'
        }
    },
    data () {
        return {
            /** @type {string|null} The current session ID, mirrors initialSessionId. */
            sessionId: null, // Will be set from prop
            /** @type {boolean} Whether a session is active and file data is considered loaded. */
            hasUploadedFile: false, // Will be set based on initialSessionId prop
            /** @type {Array<Object>} List of chat messages. */
            messages: [],
            /** @type {string} The current message being typed by the user. */
            currentMessage: '',
            /** @type {boolean} Whether the assistant is currently typing/processing. */
            isTyping: false,
            /** @type {string|null} Error message related to chat operations, not initial upload. */
            chatError: null, // Changed from uploadError
            /** @type {string} The base URL for the backend API. */
            backendUrl: globalConfig.dev.proxyTable['/api/upload'].target, // Corrected: Points to http://localhost:8000
            /** @type {string} The selected format for downloading chat history. */
            downloadFormat: 'text',
            /** @type {boolean} Whether to show the quick questions dropdown. */
            showQuickQuestions: false,
            /** @type {WebSocket|null} The WebSocket connection instance. */
            websocket: null,
            /** @type {boolean} Flag to indicate if WebSocket should be used. Falls back to HTTP if false. */
            useWebSocket: true, // WebSocket support is now implemented
            /** @type {Object|null} Holds the currently streaming message from WebSocket. */
            streamingMessage: null,
            /** @type {Array<string>} List of predefined quick questions. */
            quickQuestions: [
                'What was the highest altitude reached during the flight?',
                'When did the GPS signal first get lost?',
                'What was the maximum battery temperature?',
                'How long was the total flight time?',
                'List all critical errors that happened mid-flight.',
                'When was the first instance of RC signal loss?',
                'Are there any anomalies in this flight?',
                'Can you spot any issues in the GPS data?',
                'What was the average speed during the flight?',
                'Show me a summary of this flight log.',
                'What was the battery voltage at landing?',
                'Were there any compass calibration issues?',
                'What was the maximum distance from home point?'
            ],
            // Constants for template access
            MESSAGE_ROLES,
            validFileExtensions: VALID_FILE_EXTENSIONS
        }
    },
    computed: {
    /**
     * Returns a shortened version of the session ID for display.
     * @returns {string}
     */
        sessionIdShort () {
            return this.sessionId ? this.sessionId.substring(0, 8) : ''
        },
        /**
     * Returns a string listing supported file formats for display.
     * @returns {string}
     */
        supportedFormatsString () {
            return VALID_FILE_EXTENSIONS.map(ext => ext.substring(1).toUpperCase()).join(', ')
        },
        /**
     * Provides WebSocket class for template checks - useful if WebSocket global is not available in template directly.
     * @returns {Object}
     */
        WebSocket () {
            return WebSocket
        }
    },
    watch: {
        initialSessionId: {
            immediate: true,
            handler (newSessionId) {
                if (newSessionId) {
                    this.activateSession(newSessionId, this.initialFileName)
                } else {
                    this.deactivateSession()
                }
            }
        }
    },
    beforeDestroy () {
        this.disconnectWebSocket()
    },
    methods: {
    /**
     * Handles the chat button click - establishes WebSocket connection and sends message.
     * This is called when the user clicks the Send button.
     */
        async onChatButtonClick () {
            // Establish WebSocket connection if enabled and not already connected
            const needsConnection = !this.websocket || this.websocket.readyState !== WebSocket.OPEN
            if (this.useWebSocket && this.sessionId && needsConnection) {
                await this.connectWebSocket()
            }
            // Then send the message
            await this.sendMessage()
        },

        /**
     * Activates the chat session when a session ID is provided.
     * @param {string} sessionId - The session ID.
     * @param {string} fileName - The name of the associated file.
     */
        async activateSession (sessionId, fileName) {
            this.sessionId = sessionId
            this.hasUploadedFile = true
            this.messages = [] // Clear any old messages
            this.currentMessage = ''
            this.isTyping = false
            this.chatError = null

            // First try to fetch existing messages for this session
            try {
                await this.fetchExistingMessages()
            } catch (error) {
                console.error('Failed to fetch existing messages:', error)
                // No system messages added - removed as requested
            }

            // WebSocket connection will be established on first message
            // Not connecting here to save resources until user actually uses chat
        },

        /**
         * Fetches existing messages for the current session from the backend
         */
        async fetchExistingMessages () {
            if (!this.sessionId) return

            try {
                const response = await axios.get(`${this.backendUrl}/get_messages/${this.sessionId}`)

                if (response.data && Array.isArray(response.data)) {
                    // If we have messages, use them instead of adding default system messages
                    if (response.data.length > 0) {
                        this.messages = response.data.map(msg => ({
                            id: msg.message_id || Date.now() + Math.random(),
                            role: msg.role,
                            content: msg.content,
                            timestamp: new Date(msg.timestamp),
                            isStreaming: false,
                            analysis: msg.metadata || null
                        }))
                    } else {
                        // No default system messages - they've been removed as requested
                    }
                }
            } catch (error) {
                console.error('Error fetching messages:', error)
                throw error
            }
        },

        /**
     * Deactivates the current session, e.g., if initialSessionId becomes null.
     */
        deactivateSession () {
            this.resetSessionState(false) // Don't add system message as it implies user action
        },

        /**
     * Sends the current message from the input field.
     * Handles WebSocket connection and fallback to HTTP.
     */
        async sendMessage () {
            const trimmedMessage = this.currentMessage.trim()
            if (!trimmedMessage || this.isTyping || !this.hasUploadedFile) return

            // Clear the input field
            this.currentMessage = ''

            // Add user message to the chat immediately
            this.addMessage(MESSAGE_ROLES.USER, trimmedMessage)
            this.scrollToBottom()

            // Show typing indicator while waiting for response
            this.isTyping = true

            // Use WebSocket if enabled
            if (this.useWebSocket && this.sessionId) {
                // WebSocket connection should already be established when the Send button is clicked
                // But check just in case and connect if needed
                if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                    await this.connectWebSocket()
                }

                const wsSuccess = await this.sendMessageWebSocket(trimmedMessage)
                if (wsSuccess) return
            }

            // Fall back to HTTP if WebSocket fails
            await this._sendHttpMessage(trimmedMessage)
        },

        /**
     * Internal method to send a message via HTTP.
     * @param {string} messageText - The text of the message to send.
     */
        async _sendHttpMessage (messageText) {
            try {
                // Prepare the request data with snake_case keys for backend API
                const requestData = {
                    message: messageText,
                    // eslint-disable-next-line camelcase
                    session_id: this.sessionId, // Backend expects snake_case
                    // eslint-disable-next-line camelcase
                    message_id: Date.now().toString(), // Backend expects snake_case
                    // eslint-disable-next-line camelcase
                    max_tokens: 4000 // Backend expects snake_case
                }

                console.log('Sending chat request:', requestData)

                // Make the API call to the backend
                const response = await axios.post(`${this.backendUrl}/chat`, requestData, {
                    headers: {
                        'Content-Type': 'application/json'
                    }
                })

                console.log('Received chat response:', response.data)

                // Turn off typing indicator
                this.isTyping = false

                // Extract the response from ChatResponse
                if (response.data && response.data.status === 'SUCCESS') {
                    // Add the assistant's response to the chat
                    this.addMessage(
                        MESSAGE_ROLES.ASSISTANT,
                        response.data.message,
                        response.data.metadata
                    )

                    // Update session ID if needed
                    if (response.data.session_id && response.data.session_id !== this.sessionId) {
                        this.sessionId = response.data.session_id
                        this.addSystemMessage('Session ID updated by server.')
                    }
                } else {
                    // Handle error response
                    const errorMessage = response.data?.message ||
                        'Sorry, I encountered an error processing your request.'
                    this.addMessage(MESSAGE_ROLES.ASSISTANT, errorMessage)
                    this.chatError = errorMessage
                }
            } catch (error) {
                // Handle network or other errors
                this.isTyping = false
                console.error('Chat error (HTTP):', error.response || error.message)

                const errorDetail = error.response?.data?.detail || error.response?.data?.message
                const errorMessage = errorDetail ||
                    'Sorry, I encountered an error processing your request via HTTP.'

                this.addMessage(MESSAGE_ROLES.ASSISTANT, errorMessage)
                this.chatError = errorMessage
            }

            // Always scroll to bottom after adding a new message
            this.scrollToBottom()
        },

        /**
     * Populates the input field with a quick question and sends it.
     * @param {string} question - The quick question to ask.
     */
        askQuickQuestion (question) {
            this.currentMessage = question
            this.showQuickQuestions = false // Close the quick questions dropdown
            this.sendMessage()
        },

        /**
         * Toggles the visibility of quick questions dropdown.
         */
        toggleQuickQuestions () {
            this.showQuickQuestions = !this.showQuickQuestions
            console.log('Quick Questions toggled:', this.showQuickQuestions)
        },

        /**
     * Adds a message to the chat history.
     * @param {string} role - The role of the message sender (user, assistant, system).
     * @param {string} content - The content of the message.
     * @param {Object|null} [analysis=null] - Optional analysis data associated with the message.
     */
        addMessage (role, content, analysis = null) {
            const message = {
                id: Date.now() + Math.random(),
                role,
                content,
                analysis,
                timestamp: new Date(),
                isStreaming: false
            }
            this.messages.push(message)
            this.scrollToBottom()
        },

        /**
     * Adds a system message to the chat.
     * @param {string} content - The content of the system message.
     */
        addSystemMessage (content) {
            this.addMessage(MESSAGE_ROLES.SYSTEM, content)
        },

        /**
     * Ends the current chat session by calling the backend DELETE endpoint.
     * Then, it resets the local state and emits an event for the parent.
     */
        async endSessionAndReset () {
            if (!this.sessionId) return
            const oldSessionId = this.sessionId // Keep for logging
            try {
                await axios.delete(`${this.backendUrl}/session/${this.sessionId}`, {
                    // Note: FastAPI expects sessionId in path, not header for this specific DELETE endpoint
                    // If your backend /session/{sessionId} requires X-Session-ID header, add it here.
                })
                this.addSystemMessage('Session ended successfully.')
            } catch (error) {
                console.error(`Error ending session ${oldSessionId} on server:`, error)
                this.addSystemMessage('Failed to end session on server, but client is resetting.')
            } finally {
                this.resetSessionState(true) // Reset and indicate user ended session for potential parent handling
                this.$emit('session-ended', oldSessionId)
            }
        },

        /**
     * Clears the chat messages locally but keeps the session active on the backend.
     */
        clearChat () {
            this.messages = []
            this.addSystemMessage('Chat cleared locally. Your session is still active on the server.')
        },

        /**
     * Downloads the chat history.
     */
        async downloadChat () {
            if (!this.sessionId || this.messages.length === 0) {
                this.addSystemMessage('No chat history to download.')
                return
            }
            try {
                // Use the backend's export_chat endpoint with the specified format
                const response = await axios.get(`${this.backendUrl}/export_chat/${this.sessionId}`, {
                    params: { format: this.downloadFormat },
                    headers: { 'X-Session-ID': this.sessionId },
                    responseType: 'blob'
                })
                this._triggerBlobDownload(response.data, response.headers['content-disposition'])
                // No success message needed - the download itself is confirmation
            } catch (error) {
                console.error('Error downloading chat from server:', error)
                // Fallback for text format if direct server download fails
                if (this.downloadFormat === 'text') {
                    console.warn('Backend download failed, attempting frontend text format fallback.')
                    const chatContent = this.formatChatForFrontendDownload(this.messages)
                    const blob = new Blob([chatContent], { type: 'text/plain;charset=utf-8' })
                    this._triggerBlobDownload(blob, null, 'txt') // Pass 'txt' as the extension override
                    // No success message needed - the download itself is confirmation
                } else {
                    this.addSystemMessage(
                        'Failed to download chat history from server. ' +
                        'Please try again or select text format (.txt) for a potential fallback.'
                    )
                }
            }
        },

        /**
     * Helper to trigger a file download from a blob.
     * @param {Blob} blob - The data blob to download.
     * @param {string} [contentDispositionHeader] - Optional Content-Disposition header to extract filename.
     * @param {string} [extensionOverride] - Optional extension override to use instead of downloadFormat.
     */
        _triggerBlobDownload (blob, contentDispositionHeader, extensionOverride) {
            const url = window.URL.createObjectURL(blob)
            const link = document.createElement('a')
            link.href = url

            // Use the extension override if provided, otherwise use the downloadFormat
            const extension = extensionOverride || this.downloadFormat
            let filename =
                `uav-chat-${this.sessionIdShort}-` +
                `${new Date().toISOString().split('T')[0]}.${extension}`
            if (contentDispositionHeader) {
                const filenameMatch = contentDispositionHeader.match(/filename[^;=\\n]*=((['"]).*?\\2|[^;\\n]*)/)
                if (filenameMatch && filenameMatch[1]) {
                    filename = filenameMatch[1].replace(/['"]/g, '')
                }
            }

            link.download = filename
            document.body.appendChild(link)
            link.click()
            document.body.removeChild(link)
            window.URL.revokeObjectURL(url)
        },

        /**
     * Formats chat messages for frontend TXT download (fallback).
     * @param {Array<Object>} messagesToFormat - The messages to format.
     * @returns {string} The formatted chat content as a string.
     */
        formatChatForFrontendDownload (messagesToFormat) {
            // Simple header with session info
            const header = 'UAV Log Viewer Chat History\n' +
                         `Session ID: ${this.sessionId || 'N/A'}\n` +
                         `Generated: ${new Date().toLocaleString()}\n\n`

            let chatContent = header

            // Format each message according to the specified format: [timestamp] Role: Message
            chatContent += messagesToFormat.map(message => {
                const timestamp = new Date(message.timestamp).toLocaleString()
                const sender = message.role.charAt(0).toUpperCase() + message.role.slice(1) // Capitalize first letter
                return `[${timestamp}] ${sender}: ${message.content}\n\n`
            }).join('')

            return chatContent
        },

        /**
     * Resets the component state to initial values.
     * @param {boolean} [userInitiated=false] - Whether reset was due to direct user action (e.g. End Session button).
     */
        resetSessionState (userInitiated = false) {
            this.disconnectWebSocket()
            this.sessionId = null
            this.hasUploadedFile = false
            this.messages = []
            this.currentMessage = ''
            this.isTyping = false
            this.chatError = null
            if (userInitiated) {
                // No system message here, parent will handle UI update post session-ended event
            } else if (this.initialSessionId === null) {
                this.addSystemMessage('Session has ended.')
            }
        },

        scrollToBottom () {
            this.$nextTick(() => {
                const chatMessagesEl = this.$refs.chatMessages
                if (chatMessagesEl) {
                    chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight
                }
            })
        },

        formatTime (timestamp) {
            if (!timestamp) return ''
            const date = new Date(timestamp)
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        },

        /**
         * Formats message content for display, handling markdown and code blocks.
         * @param {string|object} content - The message content to format.
         * @returns {string} The formatted message content with HTML.
         */
        formatMessageContent (content) {
            // Handle null/undefined
            if (!content) return ''

            // Convert to string if it's not already a string
            if (typeof content !== 'string') {
                try {
                    // Try to convert objects to JSON string
                    content = JSON.stringify(content)
                } catch (e) {
                    // If stringify fails, use toString or empty string as fallback
                    content = content.toString ? content.toString() : ''
                }
            }

            // Now that we're sure content is a string, apply formatting
            try {
                // Simple markdown-like formatting
                return content
                    // Convert code blocks with ```
                    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
                    // Convert inline code with `
                    .replace(/`([^`]+)`/g, '<code>$1</code>')
                    // Convert bold with **
                    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                    // Convert italic with *
                    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
                    // Convert URLs to links
                    .replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank">$1</a>')
                    // Convert newlines to <br>
                    .replace(/\n/g, '<br>')
            } catch (e) {
                console.error('Error formatting message content:', e)
                return String(content) // Return as plain string if formatting fails
            }
        },

        formatResponse (text) {
            if (typeof text !== 'string') return ''
            return text
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold
                .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italics
                .replace(/\n/g, '<br>') // Newlines
        },

        formatAnalysisKey (key) {
            return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
        },

        /**
     * Establishes a WebSocket connection.
     * @returns {Promise<void>} Resolves on successful connection, rejects on error or timeout.
     */
        connectWebSocket () {
            if (!this.sessionId || !this.useWebSocket) {
                if (!this.useWebSocket) console.log('WebSocket usage is disabled.')
                return
            }

            // Ensure any existing connection is closed before opening a new one
            if (this.websocket && this.websocket.readyState !== WebSocket.CLOSED) {
                this.websocket.close()
            }

            const wsUrl = `${this.backendUrl.replace(/^http/, 'ws')}/ws/${this.sessionId}`
            console.log('Attempting to connect to WebSocket:', wsUrl)
            // System message removed as requested

            try {
                this.websocket = new WebSocket(wsUrl)

                this.websocket.onopen = () => {
                    console.log('WebSocket connection established for session:', this.sessionId)
                    // System message removed as requested
                    this.isTyping = false // In case it was set by a previous action
                    // If there was a chatError from HTTP fallback or previous attempt, clear it
                    this.chatError = null
                }

                this.websocket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data)
                        this.handleWebSocketMessage(data)
                    } catch (e) {
                        console.error('Error parsing WebSocket message:', e)
                        this.addMessage(MESSAGE_ROLES.SYSTEM, 'Received malformed message from server.')
                    }
                }

                this.websocket.onclose = (event) => {
                    console.log(`WebSocket disconnected. Code: ${event.code}, Reason: ${event.reason || 'N/A'}`)
                    if (event.code !== 1000 && event.code !== 1005) { // 1000 normal, 1005 no status
                        console.warn('WebSocket closed unexpectedly. May need to fall back to HTTP if still active.')
                        // this.useWebSocket = false; // Consider context before disabling globally.
                    }
                    this.websocket = null
                }

                this.websocket.onerror = (error) => {
                    console.error('WebSocket connection error:', error)
                    // console.warn('Disabling WebSocket due to error, will use HTTP fallback.');
                    if (this.websocket) this.websocket.close()
                    this.websocket = null
                    this.chatError = 'WebSocket connection error. Please try standard method.'
                }
            } catch (error) {
                console.error('WebSocket connection error:', error)
                this.chatError = 'WebSocket connection error. Please try standard method.'
            }
        },

        /**
     * Disconnects the WebSocket connection if it exists.
     */
        disconnectWebSocket () {
            if (this.websocket) {
                console.log('Disconnecting WebSocket.')
                this.websocket.onopen = null
                this.websocket.onmessage = null
                this.websocket.onclose = null
                this.websocket.onerror = null
                this.websocket.close(1000, 'Client initiated disconnect')
                this.websocket = null
            }
        },

        /**
     * Handles incoming messages from the WebSocket.
     * @param {Object} data - The parsed message data.
     */
        handleWebSocketMessage (data) {
            if (data.type === 'stream_start') {
                // Create a new streaming message placeholder
                this.streamingMessage = {
                    id: data.message_id,
                    role: MESSAGE_ROLES.ASSISTANT,
                    content: '',
                    timestamp: new Date(),
                    isStreaming: true
                }
                this.messages.push(this.streamingMessage)
                this.scrollToBottom()
            } else if (data.type === 'stream_token' && this.streamingMessage) {
                // Append token to the streaming message
                this.streamingMessage.content += data.token
                this.scrollToBottom()
            } else if (data.type === 'stream_end' && this.streamingMessage) {
                // Finalize the streaming message
                this.streamingMessage.isStreaming = false

                // Add analysis data if available from the response
                if (data.response && data.response.metadata) {
                    this.streamingMessage.analysis = data.response.metadata
                }

                // Reset streaming message reference
                this.streamingMessage = null
                this.isTyping = false
            } else if (data.type === 'error') {
                // Handle error messages
                console.error('WebSocket error:', data.message)
                this.chatError = data.message
                this.isTyping = false

                // If we were streaming, mark the message as complete
                if (this.streamingMessage) {
                    this.streamingMessage.isStreaming = false
                    this.streamingMessage.content += `\n\nError: ${data.message}`
                    this.streamingMessage = null
                } else {
                    // Add an error message if we weren't streaming
                    this.addSystemMessage(`Error: ${data.message}`)
                }

                // Log additional error details if available
                if (data.response) {
                    console.error('Error details:', data.response)
                }
            } else if (data.type === 'message') {
                // Handle complete messages (non-streaming)
                this.addMessage(
                    data.role || MESSAGE_ROLES.ASSISTANT,
                    data.content || data.message || '',
                    data.analysis || data.metadata
                )
                this.scrollToBottom()
            } else {
                console.warn('Unknown WebSocket message type:', data.type)
            }
        },

        /**
     * Sends a message via WebSocket.
     * @param {string} messageText - The text of the message to send.
     * @returns {Promise<boolean>} True if sent successfully, false otherwise.
     */
        async sendMessageWebSocket (messageText) {
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                // Try to establish connection if not already connected
                if (!this.websocket || this.websocket.readyState === WebSocket.CLOSED) {
                    try {
                        await this.connectWebSocket()
                        // Wait a short time for connection to establish
                        await new Promise(resolve => setTimeout(resolve, 500))
                    } catch (e) {
                        console.error('Failed to establish WebSocket connection:', e)
                        return false
                    }
                }
                // If still not connected after attempt, fall back
                if (
                    !this.websocket ||
                    this.websocket.readyState !== WebSocket.OPEN
                ) {
                    console.warn('WebSocket connection failed or unavailable after connect attempt. ' +
                        'Falling back for this message.')
                    return false
                }
            }

            try {
                // Show typing indicator while waiting for response
                this.isTyping = true

                // Prepare message in the format expected by the backend
                const message = {
                    type: 'message',
                    message: messageText,
                    // eslint-disable-next-line camelcase
                    session_id: this.sessionId, // Backend expects snake_case
                    // eslint-disable-next-line camelcase
                    message_id: Date.now().toString() // Generate a unique message ID
                }

                console.log('Sending WebSocket message:', message)

                // Check WebSocket state one more time before sending
                if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                    this.websocket.send(JSON.stringify(message))
                    return true
                } else {
                    console.warn('WebSocket not in OPEN state before sending. Falling back to HTTP.')
                    return false
                }
            } catch (e) {
                console.error('Error sending WebSocket message:', e)
                this.isTyping = false
                return false
            }
        },

        async fetchChatHistory (sessionId) {
            if (!sessionId) return
            this.isTyping = true
            this.chatError = null
            try {
                const response = await axios.get(
                    `${this.backendUrl}/session/${sessionId}/history`
                )
                if (response.data && response.data.history) {
                    this.messages = response.data.history.map(item => ({
                        id: item.id || Date.now() + Math.random(), // Ensure unique ID
                        role: item.role,
                        content: item.content,
                        timestamp: item.timestamp || new Date().toISOString(),
                        analysis: item.analysis || {}
                    }))
                    this.scrollToBottom()
                    const format = this.downloadFormat
                    const blob = new Blob([response.data], { type: 'application/octet-stream' })
                    const link = document.createElement('a')
                    link.href = URL.createObjectURL(blob)
                    const filename =
                        response.headers['content-disposition']?.split('filename=')[1]?.replace(/"/g, '') ||
                        `chat_history_${sessionId.substring(0, 8)}.${format}`
                    link.setAttribute('download', filename)
                    document.body.appendChild(link)
                    link.click()
                    document.body.removeChild(link)
                    URL.revokeObjectURL(link.href)
                    // No success message needed - the download itself is confirmation
                    return
                }
            } catch (error) {
                console.error('Server-side download failed:', error)
                this.addSystemMessage(
                    'Failed to download chat history from server. ' +
                    'Please try again or select TXT format for a potential fallback.'
                )
                // Fallback to local generation if server fails for these types is not implemented to keep it simple
            } finally {
                this.isTyping = false
            }
        }
    }
}
</script>

<style scoped>
/* Chat Container */
.quick-questions-container {
  width: 100%;
  margin-bottom: 10px;
  position: relative;
}

.custom-quick-questions-button {
  width: 100%;
  padding: 10px 15px;
  background-color: #4A90E2;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
  font-weight: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.custom-quick-questions-button:hover {
  background-color: #3A80D2;
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.custom-quick-questions-button:disabled {
  background-color: #a0b4d0;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.quick-questions-list {
  position: absolute;
  bottom: 100%;
  left: 0;
  width: 100%;
  background-color: white;
  border-radius: 8px;
  z-index: 10;
  box-shadow: 0 6px 16px rgba(0,0,0,0.15);
  max-height: 400px;
  overflow-y: auto;
  margin-bottom: 8px;
  color: #333;
  border: 1px solid #e0e0e0;
}

.quick-questions-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 15px;
  background-color: #f5f8ff;
  border-bottom: 1px solid #e0e0e0;
  font-weight: 600;
  color: #4A90E2;
  position: sticky;
  top: 0;
  z-index: 1;
}

.close-quick-questions {
  background: none;
  border: none;
  color: #999;
  cursor: pointer;
  font-size: 16px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  transition: all 0.2s ease;
}

.close-quick-questions:hover {
  background-color: #f0f0f0;
  color: #555;
}

.quick-question-item {
  padding: 10px 15px;
  cursor: pointer;
  border-bottom: 1px solid #eee;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 8px;
}

.quick-question-item:hover {
  background-color: #f0f7ff;
  color: #3A80D2;
}

.quick-question-item:last-child {
  border-bottom: none;
}

.quick-question-item i {
  color: #4A90E2;
  font-size: 14px;
}

/* General container styling */
.agentic-chat-container {
  height: 130vh; /* Increased height to extend further down */
  max-height: 100%; /* Ensure it doesn't overflow the container */
  display: flex;
  flex-direction: column;
  background: #2c3e50; /* Blue color matching sidebar */
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen-Sans,
               Ubuntu, Cantarell, "Helvetica Neue", sans-serif;
  color: #fff;
  overflow: auto; /* Changed to auto to allow scrolling if needed */
  font-size: 0.9rem; /* Base font size reduction */
  position: relative; /* Ensure positioning context */
}

/* Header styling */
.chat-header {
  background: #34495e; /* Darker blue for header */
  padding: 0.5rem 0.8rem; /* Reduced padding */
  border-bottom: 1px solid #4a6785;
  flex-shrink: 0;
  z-index: 10;
}

.header-action-buttons {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
  padding: 0 1rem;
}

.chat-header h3 {
  margin: 0;
  color: #4A90E2;
  font-size: 1rem; /* Reduced font size */
  font-weight: 600;
}
.chat-header h3 i {
  margin-right: 0.3rem;
}

.connection-status {
  display: flex;
  align-items: center;
}

.status-indicator {
  font-size: 0.65rem; /* Reduced font size */
  padding: 0.2rem 0.4rem; /* Reduced padding */
  border-radius: 4px;
  font-weight: 500;
}

.ws-connected {
  background-color: #50E3C2;
  color: white;
}
.ws-connected i { color: white; }

.ws-disconnected {
  background-color: #e0e0e0;
  color: #555;
}

.session-info {
  margin-top: 0.2rem;
  font-size: 0.7rem; /* Reduced font size */
  color: #bdc3c7;
  text-align: center;
}

/* Chat Interface Styling */
.chat-interface {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  min-height: 300px; /* Ensure minimum height */
  overflow: auto; /* Allow scrolling */
  background: #2c3e50; /* Blue background */
}

.chat-interface.no-file-loaded {
  justify-content: center;
  align-items: center;
}

.initial-prompt-area-fallback {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 1rem; /* Reduced padding */
  text-align: center;
  color: #777;
}

.initial-prompt-area-fallback h5 {
  font-size: 1.1rem; /* Reduced font size */
  margin-bottom: 0.5rem;
}
.initial-prompt-area-fallback h5 i {
  margin-right: 0.3rem;
}

.initial-prompt-area-fallback p {
  font-size: 0.85rem; /* Reduced font size */
  max-width: 300px;
}

.chat-messages {
  flex-grow: 1;
  overflow-y: auto;
  padding: 0.8rem; /* Reduced padding */
  scroll-behavior: smooth;
}
.chat-interface:not(.no-file-loaded) .chat-messages {
    height: 100%;
}

.message-container {
  display: flex;
  margin-bottom: 10px;
  padding: 0 10px;
}

.message-container.user {
  justify-content: flex-end;
}

.message-container.assistant {
  justify-content: flex-start;
}

.message-role {
  font-weight: bold;
  margin-right: 5px;
}

.user-message .message-role {
  color: #075e54; /* WhatsApp dark green */
}

.assistant-message .message-role {
  color: #128c7e; /* WhatsApp teal */
}

.message-content {
  border-radius: 8px;
  padding: 10px;
  max-width: 80%;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
  position: relative;
  overflow-wrap: break-word;
  word-wrap: break-word;
  word-break: break-word;
}

.user-message .message-content {
  background-color: #dcf8c6; /* WhatsApp user message color */
  margin-left: auto;
  margin-right: 10px;
}

.assistant-message .message-content {
  background-color: transparent; /* Remove background */
  margin-right: auto;
  margin-left: 10px;
  border: none; /* Remove border */
  box-shadow: none; /* Remove shadow */
}

.message-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 5px;
}

.message-body {
  font-size: 0.85rem;
}

.message-time {
  font-size: 0.65rem;
  opacity: 0.7;
}

/* Typing Indicator Styling */
.typing.assistant-message .message-content {
  padding: 0.4rem 0.6rem; /* Reduced padding */
  background: transparent; /* Match the transparent background */
}
.typing-indicator {
  display: flex;
  align-items: center;
  gap: 4px; /* Reduced gap */
  height: 18px; /* Reduced height */
}

.typing-indicator span {
  width: 6px; /* Reduced size */
  height: 6px; /* Reduced size */
  border-radius: 50%;
  background: #4A90E2;
  animation: typingDots 1.5s infinite ease-in-out;
}
/* animation is fine */

/* Chat Input Section Styling */
.chat-input-section {
  background: #34495e;
  padding: 0.6rem 0.8rem; /* Reduced padding */
  border-top: 1px solid #4a6785;
  flex-shrink: 0;
  position: sticky; /* Keep input visible */
  bottom: 0; /* Stick to bottom */
  z-index: 10; /* Ensure it stays on top */
  width: 100%;
  box-sizing: border-box;
}

.chat-input-section .input-group {
  display: flex;
}

.chat-input-section .form-control {
  flex-grow: 1;
  border: 1px solid #4a6785;
  border-right: none;
  border-radius: 6px 0 0 6px;
  padding: 0.5rem 0.8rem; /* Reduced padding */
  font-size: 0.85rem; /* Reduced font size */
  background: #ecf0f1;
  color: #2c3e50;
}
.chat-input-section .form-control:focus {
  box-shadow: none;
  border-color: #4A90E2;
  z-index: 1;
}

.chat-input-section .btn {
  background: #4A90E2;
  color: white;
  border: 1px solid #4A90E2;
  border-radius: 0 6px 6px 0;
  padding: 0.5rem 1rem; /* Adjusted padding */
  cursor: pointer;
}
.chat-input-section .btn:hover { background: #3a7bc8; }
.chat-input-section .btn:disabled {
  background: #a0c3ed;
  border-color: #a0c3ed;
}
.chat-input-section .btn i {
  font-size: 0.9rem; /* Reduced icon size */
}

.quick-questions {
  margin-top: 0.5rem; /* Reduced margin */
}
.quick-questions small {
  font-size: 0.7rem; /* Reduced font size */
  margin-bottom: 0.3rem;
}
.quick-question-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem; /* Reduced gap */
}
.quick-question-buttons .btn {
  font-size: 0.7rem; /* Reduced font size */
  padding: 0.4rem 0.7rem; /* Slightly increased padding */
  background: #3498db;
  color: white;
  border: 1px solid #2980b9;
  border-radius: 6px;
  margin-bottom: 0.5rem;
  transition: all 0.2s ease;
}
.quick-question-buttons .btn:hover {
  background: #2980b9;
  color: white;
  transform: translateY(-2px);
  box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}
/* :disabled styles are fine */

/* Session Actions Footer Styling */
.session-actions {
  background: #34495e;
  padding: 0.5rem 0.8rem; /* Reduced padding */
  border-top: 1px solid #4a6785;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 0.5rem; /* Reduced gap */
  flex-shrink: 0;
  position: sticky; /* Keep footer visible */
  bottom: 0; /* Stick to bottom */
  z-index: 9; /* Below input but above content */
}

.session-info-actions {
  font-size: 0.75rem; /* Reduced font size */
  color: #bdc3c7;
}

.action-buttons {
  display: flex;
  align-items: center;
  justify-content: space-between; /* Spread items across the container */
  width: 100%; /* Take full width */
  flex-wrap: wrap;
}

.download-section {
  width: 100%; /* Match the width of chat-input-section */
  box-sizing: border-box;
}

.download-section .input-group {
  display: flex;
  align-items: center;
}
.download-section .btn,
.action-buttons .btn-outline-secondary,
.action-buttons .btn-outline-danger {
  padding: 0.3rem 0.6rem; /* Reduced padding */
  font-size: 0.75rem; /* Reduced font size */
  border-radius: 4px;
}

.download-section .form-control {
  font-size: 0.75rem; /* Reduced font size */
  border-radius: 4px;
  border: 1px solid #e0e0e0;
  padding: 0.3rem 0.5rem; /* Reduced padding */
  margin-left: -1px;
  max-width: 100px; /* Reduced max-width */
}

/* Hover/active states for buttons are fine */

/* Custom input styling */
.custom-input-group {
  display: flex;
  width: 100%;
  margin-bottom: 0.5rem;
}

/* Quick Questions button styling to match input width */
.custom-quick-questions-button {
  width: 100%;
  padding: 0.5rem 1rem;
  background-color: #4A90E2;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 0.9rem;
  transition: background-color 0.2s;
}

.custom-quick-questions-button:hover {
  background-color: #3A7BC8;
}

.custom-quick-questions-button:disabled {
  background-color: #95a5a6;
  cursor: not-allowed;
}

.custom-input-group {
  display: flex;
  width: 100%;
  margin-bottom: 0.5rem;
  box-sizing: border-box;
}

.custom-form-control {
  flex: 1;
  height: 38px;
  padding: 0.375rem 0.75rem;
  font-size: 0.9rem;
  border: 1px solid #4a6785;
  border-radius: 6px 0 0 6px;
  background: #ecf0f1;
  color: #2c3e50;
}

.custom-form-control:focus {
  outline: none;
  border-color: #3498db;
  box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
}

.custom-send-button {
  height: 38px;
  width: 80px;
  padding: 0.375rem 0.75rem;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  border-radius: 0 6px 6px 0;
  background-color: #337ab7; /* Bootstrap primary button color */
  color: white;
  font-weight: 500;
  transition: all 0.2s ease;
  cursor: pointer;
}

.custom-send-button:hover:not(:disabled) {
  background-color: #2980b9;
  transform: translateY(-1px);
  box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}

.custom-send-button:disabled {
  background-color: #95a5a6;
  cursor: not-allowed;
  opacity: 0.7;
}

/* Style the quick questions dropdown */
.quick-questions {
  margin-top: 0.8rem;
  position: relative;
}

.quick-questions-dropdown {
  position: relative;
}

.quick-questions-dropdown .dropdown-toggle {
  background-color: #337ab7; /* Bootstrap primary button color */
  border-color: #2e6da4;
  color: white;
  width: 100%;
  text-align: center;
  padding: 0.4rem 1rem;
}

.quick-questions-text {
  margin: 0 0.5rem;
  font-weight: bold;
}

.quick-questions-dropdown .dropdown-toggle:hover {
  background-color: #2980b9;
}

/* Custom dropdown styles */
.custom-dropdown {
  position: absolute;
  width: 300px;
  background: #34495e;
  border: 1px solid #4a6785;
  border-radius: 6px;
  padding: 0.5rem;
  margin-top: 0.5rem;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  max-height: 300px;
  overflow-y: auto;
  box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}

.question-item {
  padding: 8px 12px;
  cursor: pointer;
  color: white;
  border-radius: 4px;
  transition: background-color 0.2s;
}

.question-item:hover {
  background-color: #2980b9;
}

.quick-question-list button {
  text-align: left;
  white-space: normal;
  background: #3498db;
  color: white;
  border: 1px solid #2980b9;
}

.quick-question-list button:hover {
  background: #2980b9;
  transform: translateY(-1px);
}

/* Custom download group styling */
.custom-download-group {
  display: flex;
  margin-bottom: 0.5rem;
  width: 100%; /* Match the width of custom-input-group */
}

.custom-download-btn {
  height: 38px;
  width: 254px; /* Adjusted to match chat-input-section width (334px - 80px format select) */
  padding: 0.375rem 0.75rem;
  background-color: #3498db;
  color: white;
  border: none;
  border-radius: 6px 0 0 6px;
  font-weight: 500;
  transition: all 0.2s ease;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}

.custom-download-btn:hover:not(:disabled) {
  background-color: #2980b9;
  transform: translateY(-1px);
  box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}

.custom-download-btn:disabled {
  background-color: #95a5a6;
  cursor: not-allowed;
  opacity: 0.7;
}

.custom-format-select {
  height: 38px;
  width: 80px; /* Same width as send button */
  padding: 0.375rem 0.75rem;
  border: none;
  border-radius: 0 6px 6px 0;
  background-color: #ecf0f1;
  color: #2c3e50;
  font-size: 0.9rem;
  cursor: pointer;
}

.custom-format-select:focus {
  outline: none;
  box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
}

/* Responsive adjustments for embedded context (might not be strictly necessary if sidebar width is fixed) */
/* Consider removing or simplifying @media queries if the embedded panel has a fairly static width */

/* @media (max-width: 768px) { ... } */ /* These might be too broad now */
/* @media (max-width: 480px) { ... } */

/* Example: if embedded in a narrow sidebar, always use compact styles */
.chat-header h3 { font-size: 1rem; }
.message { max-width: 90%; } /* Allow messages to take a bit more width in narrow panel */
.message-avatar { width: 24px; height: 24px; font-size: 0.7rem; }
.message-content { padding: 0.4rem 0.6rem; }
.quick-question-buttons .btn { font-size: 0.65rem; padding: 0.2rem 0.4rem; }
.action-buttons .btn, .download-section .btn { font-size: 0.7rem; padding: 0.25rem 0.5rem; }

</style>
