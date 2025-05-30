<template>
    <div>
        <li  v-if="file==null && !sampleLoaded && !showAgenticChoice" >
            <a @click="onLoadSample('sample')" class="section"><i class="fas fa-play"></i>  Open Sample </a>
        </li>
        <li v-if="url && !showAgenticChoice">
            <a @click="share" class="section"><i class="fas fa-share-alt"></i> {{ shared ? 'Copied to clipboard!' :
                'Share link'}}</a>
        </li>
        <li v-if="url && !showAgenticChoice">
            <a :href="'/uploaded/' + url" class="section" target="_blank"><i class="fas fa-download"></i> Download</a>
        </li>
        <div @click="browse" @dragover.prevent @drop="onDrop" id="drop_zone"
        v-if="file==null && uploadpercentage===-1  && !sampleLoaded && !showAgenticChoice">
            <p>Drop *.tlog or *.bin file here or click to browse</p>
            <input @change="onChange" id="choosefile" style="opacity: 0;" type="file">
        </div>

        <!-- Agentic Choice Buttons -->
        <div v-if="showAgenticChoice" class="agentic-choice-container">
            <p>Log file uploaded successfully. How would you like to proceed?</p>
            <button @click="handleStandardLogView" class="btn btn-secondary btn-sm mr-2">View Standard Log</button>
            <button @click="handleAgenticChatStart" class="btn btn-primary btn-sm">Start Agentic Analysis</button>
        </div>

        <!--<b-form-checkbox @change="uploadFile()" class="uploadCheckbox" v-if="file!=null && !uploadStarted"> Upload
        </b-form-checkbox>-->
        <VProgress v-bind:complete="transferMessage"
                   v-bind:percent="uploadpercentage"
                   v-if="uploadpercentage > -1 && !showAgenticChoice">
        </VProgress>
        <VProgress v-bind:complete="state.processStatus"
                   v-bind:percent="state.processPercentage"
                   v-if="state.processPercentage > -1 && !showAgenticChoice"
        ></VProgress>
    </div>
</template>
<script>
import VProgress from './SideBarFileManagerProgressBar.vue'
import Worker from '../tools/parsers/parser.worker.js'
import { store } from './Globals'

import { MAVLink20Processor as MAVLink } from '../libs/mavlink'

const worker = new Worker()

worker.addEventListener('message', function (event) {
})

export default {
    name: 'Dropzone',
    data: function () {
        return {
            // eslint-disable-next-line no-undef
            mavlinkParser: new MAVLink(),
            uploadpercentage: -1,
            sampleLoaded: false,
            shared: false,
            url: null,
            transferMessage: '',
            state: store,
            file: null,
            uploadStarted: false,
            showAgenticChoice: false,
            pendingSessionData: null
        }
    },
    created () {
        this.$eventHub.$on('loadType', this.loadType)
        this.$eventHub.$on('trimFile', this.trimFile)
    },
    beforeDestroy () {
        this.$eventHub.$off('open-sample')
    },
    methods: {
        trimFile () {
            worker.postMessage({ action: 'trimFile', time: this.state.timeRange })
        },
        onLoadSample (file) {
            let url
            if (file === 'sample') {
                this.state.file = 'sample'
                url = require('../assets/vtol.tlog').default
                this.state.logType = 'tlog'
            } else {
                url = file
                // Set the file name for display purposes
                const urlParts = url.split('/')
                this.state.file = urlParts[urlParts.length - 1]
            }
            const oReq = new XMLHttpRequest()
            console.log(`loading file from ${url}`)

            // Set the log type based on file extension
            this.state.logType = url.indexOf('.tlog') > 0 ? 'tlog' : 'bin'
            if (url.indexOf('.txt') > 0) {
                this.state.logType = 'dji'
            }

            oReq.open('GET', url, true)
            oReq.responseType = 'arraybuffer'

            // Use arrow function to preserve 'this' context
            oReq.onload = (oEvent) => {
                const arrayBuffer = oReq.response

                this.transferMessage = 'Download Done'
                this.sampleLoaded = true
                // For sample files, we need to construct a File object with the correct extension
                const sampleFileName = 'sample.tlog' // Explicitly set filename with extension
                this.file = new File([arrayBuffer], sampleFileName, { type: 'application/octet-stream' })
                this.state.file = sampleFileName // Also update state.file if it's used for display elsewhere

                worker.postMessage({
                    action: 'parse',
                    file: arrayBuffer,
                    isTlog: (url.indexOf('.tlog') > 0),
                    isDji: (url.indexOf('.txt') > 0)
                })
                this.uploadFile() // Call uploadFile for sample
            }
            oReq.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    this.uploadpercentage = 100 * e.loaded / e.total
                }
            }
            , false)
            oReq.onerror = (error) => {
                alert('unable to fetch remote file, check CORS settings in the target server')
                console.log(error)
            }

            oReq.send()
        },
        onChange (ev) {
            const fileinput = document.getElementById('choosefile')
            this.process(fileinput.files[0])
        },
        onDrop (ev) {
            // Prevent default behavior (Prevent file from being opened)
            ev.preventDefault()
            if (ev.dataTransfer.items) {
                // Use DataTransferItemList interface to access the file(s)
                for (let i = 0; i < ev.dataTransfer.items.length; i++) {
                    // If dropped items aren't files, reject them
                    if (ev.dataTransfer.items[i].kind === 'file') {
                        const file = ev.dataTransfer.items[i].getAsFile()
                        this.process(file)
                    }
                }
            } else {
                // Use DataTransfer interface to access the file(s)
                for (let i = 0; i < ev.dataTransfer.files.length; i++) {
                    console.log('... file[' + i + '].name = ' + ev.dataTransfer.files[i].name)
                    console.log(ev.dataTransfer.files[i])
                }
            }
        },
        loadType: function (type) {
            worker.postMessage({
                action: 'loadType',
                type: type
            })
        },
        process: function (file) {
            this.state.file = file.name
            this.state.processStatus = 'Pre-processing...'
            this.state.processPercentage = 100
            this.file = file
            const reader = new FileReader()
            reader.onload = function (e) {
                const data = reader.result
                worker.postMessage({
                    action: 'parse',
                    file: data,
                    isTlog: (file.name.endsWith('tlog')),
                    isDji: (file.name.endsWith('txt'))
                })
            }
            this.state.logType = file.name.endsWith('tlog') ? 'tlog' : 'bin'
            if (file.name.endsWith('.txt')) {
                this.state.logType = 'dji'
            }
            reader.readAsArrayBuffer(file)
            // We already have this.file = file from the beginning of this method.
            this.uploadFile() // Call uploadFile for user-provided file
        },
        uploadFile () {
            console.log('[SideBarFileManager] uploadFile called.')
            this.uploadStarted = true
            // this.transferMessage = 'Upload Done!' // Will be set based on actual response
            this.uploadpercentage = 0
            const formData = new FormData()
            if (!this.file) {
                console.error('[SideBarFileManager] uploadFile: this.file is not set!')
                this.transferMessage = 'Error: No file to upload.'
                this.uploadpercentage = 100 // End progress
                return
            }
            console.log('[SideBarFileManager] uploadFile: file to upload:', this.file.name, 'size:', this.file.size)
            formData.append('file', this.file)

            const request = new XMLHttpRequest()
            request.onload = () => {
                if (request.status >= 200 && request.status < 400) {
                    this.uploadpercentage = 100
                    try {
                        const responseData = JSON.parse(request.responseText)
                        // Keep original url if response is not JSON as expected
                        this.url = responseData.url || request.responseText

                        if (responseData.session_id) {
                            this.pendingSessionData = responseData // Store data for immediate choice
                            // Also store it globally for later activation if standard view is chosen first
                            this.state.currentLogInitialSessionId = responseData.session_id
                            this.state.currentLogInitialFileName = responseData.filename || this.file.name

                            // Debug session tracking
                            console.log(`[SideBarFileManager] Session created and stored: ${responseData.session_id}`)
                            localStorage.setItem('currentSessionId', responseData.session_id)

                            this.showAgenticChoice = true // Show choice buttons
                            this.transferMessage = 'Upload complete. Choose an option.'
                            // Do not automatically start agentic chat here
                        } else {
                            this.transferMessage = 'File processed locally (no session ID).'
                            // Proceed with standard local view if applicable or inform user.
                            // If worker parsing is separate, it will continue.
                        }
                    } catch (e) {
                        console.error('Error parsing upload response:', e)
                        this.transferMessage = 'Upload finished, but response error.'
                        this.url = request.responseText // Store raw response as URL/ID
                    }
                } else {
                    alert('error! ' + request.status)
                    this.uploadpercentage = 100
                    this.transferMessage = 'Error Uploading: ' + request.statusText
                    console.log(request)
                }
            }
            request.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    this.uploadpercentage = 100 * e.loaded / e.total
                }
            }
            , false)
            request.open('POST', '/api/upload')
            request.send(formData)
        },
        handleStandardLogView () {
            this.showAgenticChoice = false
            this.pendingSessionData = null
            this.state.showEmbeddedChat = false // Ensure embedded chat is hidden
            this.transferMessage = 'Loading standard log view...'
            // The worker process would typically continue and load the standard view.
            // If we need to explicitly trigger something, do it here.
            // Ensure the sidebar selected tab is appropriate, e.g., 'plot' if logs are processed.
            if (this.state.processDone) {
                this.$eventHub.$emit('set-selected', 'plot')
            } else {
                this.$eventHub.$emit('set-selected', 'home') // Or whatever is appropriate
            }
        },
        handleAgenticChatStart () { // Now activates the embedded chat panel
            if (this.pendingSessionData && this.pendingSessionData.session_id) {
                this.state.agenticSessionId = this.pendingSessionData.session_id
                const defaultFileName = this.file ? this.file.name : 'log file'
                this.state.agenticFileName = this.pendingSessionData.filename || defaultFileName
                // these should already be set from uploadFile, but ensure they are for chat activation
                this.state.currentLogInitialSessionId = this.state.agenticSessionId
                this.state.currentLogInitialFileName = this.state.agenticFileName

                this.state.showEmbeddedChat = true // Show the embedded chat panel
                this.state.agenticSessionActive = false // This flag is not for full-screen anymore

                this.transferMessage = 'Starting Agentic Analysis...'
                this.state.processStatus = 'Agentic Session Initialized'
                this.state.processPercentage = 100
                // Tell sidebar to select the chat view. This assumes Sidebar listens or Home relays.
                this.$eventHub.$emit('set-selected', 'chat')
            } else {
                console.error('Agentic chat start called without pending session data.')
                this.transferMessage = 'Error starting agentic chat.'
            }
            this.showAgenticChoice = false // Hide choice buttons
            this.pendingSessionData = null // Clear pending data
        },
        fixData (message) {
            if (message.name === 'GLOBAL_POSITION_INT') {
                message.lat = message.lat / 10000000
                message.lon = message.lon / 10000000
                // eslint-disable-next-line
                message.relative_alt = message.relative_alt / 1000
            }
            return message
        },
        browse () {
            document.getElementById('choosefile').click()
        },
        share () {
            const el = document.createElement('textarea')
            el.value = window.location.host + '/#/v/' + this.url
            document.body.appendChild(el)
            el.select()
            document.execCommand('copy')
            document.body.removeChild(el)
            this.shared = true
        },
        downloadFileFromURL (url) {
            const a = document.createElement('a')
            document.body.appendChild(a)
            a.style = 'display: none'
            a.href = url
            a.download = this.state.file + '-trimmed.' + this.state.logType
            a.click()
            document.body.removeChild(a)
            window.URL.revokeObjectURL(url)
        }
    },
    mounted () {
        window.addEventListener('message', (event) => {
            if (event.data.type === 'arrayBuffer') {
                worker.postMessage({
                    action: 'parse',
                    file: event.data.data,
                    isTlog: false,
                    isDji: false
                })
            }
        })
        worker.onmessage = (event) => {
            if (this.state.agenticSessionActive) {
                // If agentic chat is active, largely ignore messages from the local parser worker.
                // We might still want to observe certain events, but for now, let's just log.
                if (event.data.percentage) { // Allow progress updates for any background work
                    this.state.processPercentage = event.data.percentage
                } else {
                    console.log(
                        'Worker message received while agentic chat active, ignoring primary data:',
                        event.data.action || event.data.type || event.data
                    )
                }
                return // Important: stop further processing of this message
            }

            if (event.data.percentage) {
                this.state.processPercentage = event.data.percentage
            } else if (event.data.availableMessages) {
                this.$eventHub.$emit('messageTypes', event.data.availableMessages)
            } else if (event.data.metadata) {
                this.state.metadata = event.data.metadata
            } else if (event.data.messages) {
                this.state.messages = event.data.messages
                this.$eventHub.$emit('messages')
            } else if (event.data.messagesDoneLoading) {
                this.$eventHub.$emit('messagesDoneLoading')
            } else if (event.data.messageType) {
                this.state.messages[event.data.messageType] = event.data.messageList
                this.$eventHub.$emit('messages')
            } else if (event.data.files) {
                this.state.files = event.data.files
                this.$eventHub.$emit('messages')
            } else if (event.data.url) {
                this.downloadFileFromURL(event.data.url)
            }
        }
        const url = document.location.search.split('?file=')[1]
        if (url) {
            this.onLoadSample(decodeURIComponent(url))
        }
    },
    components: {
        VProgress
    }
}
</script>
<style scoped>

    /* NAVBAR */

    #drop_zone {
        padding-top: 25px;
        padding-left: 10px;
        border: 2px dashed #434b52da;
        width: auto;
        height: 100px;
        margin: 20px;
        border-radius: 5px;
        cursor: default;
        background-color: rgba(0, 0, 0, 0);
    }

    #drop_zone:hover {
        background-color: #171e2450;
    }

    .uploadCheckbox {
        margin-left: 20px;
    }

    .agentic-choice-container {
        padding: 20px;
        text-align: center;
        border: 1px solid #ccc;
        margin: 20px;
        border-radius: 5px;
        background-color: #f9f9f9;
    }
    .agentic-choice-container p {
        margin-bottom: 15px;
    }

</style>
