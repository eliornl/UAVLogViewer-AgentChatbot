import '../mavextra/mavextra'
import '../mavextra/mymavextra'

export const store = {
    // currentTrajectory: [],
    trajectorySource: '',
    trajectories: {},
    timeTrajectory: {},
    timeAttitude: {},
    timeAttitudeQ: {},
    logType: '',
    defaultParams: {},
    showParams: false,
    showRadio: false,
    showMessages: false,
    showDeviceIDs: false,
    showAttitude: false,
    showEkfHelper: false,
    flightModeChanges: [],
    file: null,
    events: [],
    cssColors: [],
    colors: [],
    mapAvailable: false,
    mission: [],
    fences: [],
    showFences: true,
    showMap: false,
    showMagFit: false,
    currentTime: false,
    processDone: false,
    plotOn: false,
    processStatus: 'Pre-processing...',
    processPercentage: -1,
    mapLoading: false,
    plotLoading: false,
    timeRange: null,
    textMessages: [],
    namedFloats: [],
    metadata: null,
    // cesium menu:
    modelScale: 1.0,
    heightOffset: 0.0,
    showClickableTrajectory: false,
    showTrajectory: true,
    trajectorySources: [],
    attitudeSources: {},
    attitudeSource: null,
    showWaypoints: true,
    cameraType: 'follow',
    expressions: [], // holds message name
    expressionErrors: [],
    plotCache: {},
    allAxis: [0, 1, 2, 3, 4, 5],
    allColors: [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467BD',
        '#8C564B'],
    radioMode: '2',
    /* global _COMMIT_ */
    commit: _COMMIT_.slice(0, 6),
    /* global _BUILDDATE_ */
    buildDate: _BUILDDATE_,
    childPlots: [],
    // Agentic Chat State
    agenticSessionId: null,
    agenticFileName: null,
    agenticSessionActive: false,
    isUploadingAgenticLog: false,
    agenticUploadError: null,
    showAgenticChatUpload: true, // Initially true to show upload prompt
    currentLogInitialSessionId: null, // New: Stores session_id from latest upload
    currentLogInitialFileName: null, // New: Stores filename from latest upload
    showEmbeddedChat: false, // New: Controls visibility of chat panel in sidebar
    isLoadingAgenticLog: false
}

export const config = {
    // ... existing code ...
}
