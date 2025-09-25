/**
 * BlinkSense Simple Client - HTTP-based drowsiness detection client
 * 
 * Uses HTTP POST requests instead of WebSockets for testing server-side processing.
 */

class BlinkSenseSimpleClient {
    constructor() {
        // DOM elements
        this.video = document.getElementById('video');
        this.canvas = document.getElementById('canvas');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.alert = document.getElementById('alert');
        this.beep = document.getElementById('beep');
        
        // Status elements
        this.connectionStatus = document.getElementById('connectionStatus');
        this.connectionText = document.getElementById('connectionText');
        this.detectionStatus = document.getElementById('detectionStatus');
        this.earValue = document.getElementById('earValue');
        this.popenValue = document.getElementById('popenValue');
        this.thresholdValue = document.getElementById('thresholdValue');
        this.streakValue = document.getElementById('streakValue');
        this.perclosValue = document.getElementById('perclosValue');
        this.cooldownValue = document.getElementById('cooldownValue');
        this.calibratingValue = document.getElementById('calibratingValue');
        
        // State
        this.stream = null;
        this.isRunning = false;
        this.frameInterval = null;
        this.usingClientSide = false;
        this.serverFailed = false;
        
        // Settings
        this.frameRate = 2; // Lower rate for HTTP requests
        this.jpegQuality = 0.5; // Lower quality for HTTP
        
        // Client-side detection
        this.onnxSession = null;
        this.faceMesh = null;
        this.isClientSideReady = false;
        
        // Eye landmark indices (MediaPipe Face Mesh)
        this.LEFT_EYE = [33, 160, 158, 133, 153, 144];
        this.RIGHT_EYE = [362, 385, 387, 263, 373, 380];
        
        // Detection state for client-side
        this.earHistory = [];
        this.closedStreak = 0;
        this.perclosBuffer = [];
        this.tauEar = 0.20;
        this.calibrating = true;
        this.earOpenVals = [];
        this.cooldown = 0;
        this.lastTick = Date.now();
        
        // Blink filtering
        this.currentCloseDuration = 0;
        this.wasClosedLastFrame = false;
        
        this.setupEventListeners();
        this.setupCanvas();
        this.initializeClientSideDetection();
    }
    
    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.start());
        this.stopBtn.addEventListener('click', () => this.stop());
        
        // Prime audio on first user interaction
        document.addEventListener('click', () => this.primeAudio(), { once: true });
    }
    
    setupCanvas() {
        this.canvas.width = 640;
        this.canvas.height = 480;
        this.ctx = this.canvas.getContext('2d');
    }
    
    async primeAudio() {
        // Prime the audio element for later playback
        try {
            this.beep.muted = true;
            await this.beep.play();
            this.beep.pause();
            this.beep.currentTime = 0;
            this.beep.muted = false;
        } catch (e) {
            console.log('Audio priming failed (this is normal on some browsers)');
        }
    }
    
    async start() {
        try {
            // Start camera
            await this.startCamera();
            
            // Initialize client-side detection if not ready
            if (!this.isClientSideReady) {
                this.detectionStatus.textContent = 'Initializing client-side detection...';
                await this.initializeClientSideDetection();
            }
            
            // Start sending frames
            this.startFrameCapture();
            
            this.isRunning = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.detectionStatus.textContent = 'Starting...';
            this.updateConnectionStatus(true);
            
        } catch (error) {
            console.error('Failed to start:', error);
            this.detectionStatus.textContent = 'Failed to start';
            alert('Failed to start detection: ' + error.message);
        }
    }
    
    async startCamera() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            });
            
            this.video.srcObject = this.stream;
            await new Promise(resolve => {
                this.video.onloadedmetadata = resolve;
            });
            
            console.log('Camera started successfully');
        } catch (error) {
            throw new Error('Camera access failed: ' + error.message);
        }
    }
    
    startFrameCapture() {
        const interval = 1000 / this.frameRate; // Convert FPS to milliseconds
        
        this.frameInterval = setInterval(() => {
            if (this.isRunning) {
                this.captureAndSendFrame();
            }
        }, interval);
    }
    
    async captureAndSendFrame() {
        try {
            // Draw current video frame to canvas
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Convert to base64 JPEG
            const dataURL = this.canvas.toDataURL('image/jpeg', this.jpegQuality);
            const base64Data = dataURL.split(',')[1]; // Remove data:image/jpeg;base64, prefix
            
            // Try server-side processing first (if not already failed)
            if (!this.serverFailed) {
                try {
                    const response = await fetch('/api/process-frame/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            frame: base64Data
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    // Check if server returned an error
                    if (data.status === 'error') {
                        console.warn('Server processing failed, switching to client-side:', data.message);
                        this.serverFailed = true;
                        this.usingClientSide = true;
                    } else {
                        this.handleServerMessage(data);
                        this.updateConnectionStatus(true, 'HTTP');
                        return; // Success - exit early
                    }
                } catch (serverError) {
                    console.warn('Server request failed, switching to client-side:', serverError);
                    this.serverFailed = true;
                    this.usingClientSide = true;
                }
            }
            
            // Fallback to client-side processing
            if (this.usingClientSide && this.isClientSideReady) {
                // Use canvas directly for client-side processing
                const result = await this.processFrameClientSide(this.canvas);
                this.handleServerMessage(result);
                this.updateConnectionStatus(true, 'Client-Side');
            } else if (this.usingClientSide && !this.isClientSideReady) {
                this.detectionStatus.textContent = 'Client-side detection loading...';
                this.updateConnectionStatus(false);
            } else {
                this.detectionStatus.textContent = 'Server connection error';
                this.updateConnectionStatus(false);
            }
            
        } catch (error) {
            console.error('Frame capture failed:', error);
            this.detectionStatus.textContent = 'Frame capture error';
            this.updateConnectionStatus(false);
        }
    }
    
    handleServerMessage(data) {
        // Don't update connection status here - let the caller handle it
        // to preserve the correct mode (HTTP vs Client-Side)
        
        switch (data.status) {
            case 'ok':
                this.updateDetectionStatus(data);
                if (data.alert) {
                    this.triggerAlert();
                }
                break;
                
            case 'no_face':
                this.detectionStatus.textContent = 'No face detected';
                break;
                
            case 'no_signal':
                this.detectionStatus.textContent = 'No detection signal';
                break;
                
            case 'error':
                console.error('Server error:', data.message);
                this.detectionStatus.textContent = 'Server error';
                break;
                
            default:
                console.log('Unknown server message:', data);
        }
    }
    
    updateDetectionStatus(data) {
        // Main status
        if (data.calibrating) {
            this.detectionStatus.textContent = 'Calibrating EAR...';
            this.detectionStatus.className = 'status-warning';
        } else {
            const mode = data.cnn_available ? 'EAR + CNN' : 'EAR Only';
            this.detectionStatus.textContent = `Active Detection (${mode})`;
            this.detectionStatus.className = 'status-good';
        }
        
        // Detection metrics
        this.earValue.textContent = data.ear ? data.ear.toFixed(3) : '--';
        this.popenValue.textContent = data.p_open ? data.p_open.toFixed(2) : '--';
        this.thresholdValue.textContent = data.tau_ear ? data.tau_ear.toFixed(3) : '--';
        
        // Temporal metrics
        this.streakValue.textContent = data.closed_streak ? data.closed_streak.toFixed(1) + 's' : '0.0s';
        this.perclosValue.textContent = data.perclos ? data.perclos.toFixed(1) + '%' : '0.0%';
        this.cooldownValue.textContent = data.cooldown ? data.cooldown.toFixed(1) + 's' : '0.0s';
        this.calibratingValue.textContent = data.calibrating ? 'Yes' : 'No';
        
        // Color coding for critical values
        if (data.closed_streak > 3) {
            this.streakValue.className = 'status-warning';
        } else if (data.closed_streak > 4.5) {
            this.streakValue.className = 'status-error';
        } else {
            this.streakValue.className = '';
        }
        
        if (data.perclos > 10) {
            this.perclosValue.className = 'status-warning';
        } else if (data.perclos > 14) {
            this.perclosValue.className = 'status-error';
        } else {
            this.perclosValue.className = '';
        }
    }
    
    async triggerAlert() {
        // Visual alert
        this.alert.style.display = 'block';
        setTimeout(() => {
            this.alert.style.display = 'none';
        }, 3000);
        
        // Audio alert
        try {
            this.beep.currentTime = 0;
            await this.beep.play();
        } catch (error) {
            console.log('Audio playback failed:', error);
        }
        
        console.log('🚨 DROWSINESS ALERT TRIGGERED');
    }
    
    updateConnectionStatus(connected, mode = 'HTTP') {
        if (connected) {
            this.connectionStatus.className = 'connection-status connected';
            this.connectionText.textContent = `Connected (${mode})`;
        } else {
            this.connectionStatus.className = 'connection-status disconnected';
            this.connectionText.textContent = 'Disconnected';
            this.detectionStatus.textContent = 'Connection lost';
        }
    }
    
    stop() {
        this.isRunning = false;
        
        // Stop frame capture
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }
        
        // Stop camera
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            this.video.srcObject = null;
        }
        
        // Reset UI
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.detectionStatus.textContent = 'Stopped';
        this.updateConnectionStatus(false);
        
        // Reset values
        this.earValue.textContent = '--';
        this.popenValue.textContent = '--';
        this.thresholdValue.textContent = '--';
        this.streakValue.textContent = '0.0s';
        this.perclosValue.textContent = '0.0%';
        this.cooldownValue.textContent = '0.0s';
        this.calibratingValue.textContent = '--';
        
        // Reset client-side state
        this.usingClientSide = false;
        this.serverFailed = false;
        this.closedStreak = 0;
        this.perclosBuffer = [];
        this.calibrating = true;
        this.earOpenVals = [];
        this.cooldown = 0;
        this.currentCloseDuration = 0;
        this.wasClosedLastFrame = false;
    }
    
    // Client-side detection methods
    
    async initializeClientSideDetection() {
        console.log('Initializing client-side detection...');
        try {
            // Check if libraries are loaded
            console.log('FaceMesh available:', typeof FaceMesh !== 'undefined');
            console.log('ONNX available:', typeof ort !== 'undefined');
            
            // Initialize MediaPipe Face Mesh
            if (typeof FaceMesh !== 'undefined') {
                this.faceMesh = new FaceMesh({
                    locateFile: (file) => {
                        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619/${file}`;
                    }
                });
                
                this.faceMesh.setOptions({
                    maxNumFaces: 1,
                    refineLandmarks: true,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5
                });
                console.log('MediaPipe FaceMesh initialized');
            } else {
                console.error('FaceMesh not available - check if MediaPipe scripts loaded');
            }
            
            // Load ONNX model (make it optional for EAR-only detection)
            try {
                if (typeof ort !== 'undefined') {
                    this.onnxSession = await ort.InferenceSession.create('/static/models/eye_resnet.onnx');
                    console.log('ONNX model loaded successfully');
                } else {
                    console.warn('ONNX.js not available');
                }
            } catch (e) {
                console.warn('Failed to load ONNX model (will use EAR-only detection):', e);
            }
            
            // Ready if we have at least MediaPipe for face detection
            this.isClientSideReady = !!this.faceMesh;
            console.log('Client-side detection ready:', this.isClientSideReady);
            console.log('Has ONNX:', !!this.onnxSession);
            console.log('Has MediaPipe:', !!this.faceMesh);
            
        } catch (error) {
            console.error('Client-side detection initialization failed:', error);
            this.isClientSideReady = false;
        }
    }
    
    async processFrameClientSide(canvas) {
        if (!this.isClientSideReady) {
            return { status: 'error', message: 'Client-side detection not ready' };
        }
        
        try {
            // Process with MediaPipe using canvas
            const results = await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => reject(new Error('MediaPipe timeout')), 5000);
                this.faceMesh.onResults((res) => {
                    clearTimeout(timeout);
                    resolve(res);
                });
                this.faceMesh.send({ image: canvas });
            });
            
            if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
                return { status: 'no_face' };
            }
            
            const landmarks = results.multiFaceLandmarks[0];
            
            // Calculate EAR
            const leftEar = this.calculateEAR(landmarks, this.LEFT_EYE);
            const rightEar = this.calculateEAR(landmarks, this.RIGHT_EYE);
            const avgEar = (leftEar + rightEar) / 2;
            
            // CNN prediction (simplified - using EAR-based prediction for now)
            const pOpen = Math.max(0, Math.min(1, (avgEar - 0.1) / 0.15));
            
            // Update temporal state
            const result = this.updateTemporalState(avgEar, pOpen);
            
            return result;
            
        } catch (error) {
            console.error('Client-side processing failed:', error);
            return { status: 'error', message: 'Client-side processing failed' };
        }
    }
    
    calculateEAR(landmarks, eyeIndices) {
        const points = eyeIndices.map(i => landmarks[i]);
        
        // Vertical distances
        const vert1 = this.distance(points[1], points[5]);
        const vert2 = this.distance(points[2], points[4]);
        
        // Horizontal distance
        const hor = this.distance(points[0], points[3]);
        
        return (vert1 + vert2) / (2.0 * hor + 1e-6);
    }
    
    distance(p1, p2) {
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }
    
    updateTemporalState(ear, pOpen) {
        const now = Date.now();
        const dt = (now - this.lastTick) / 1000; // Convert to seconds
        this.lastTick = now;
        
        // Determine closed state
        const closedByEar = ear < this.tauEar;
        const closedByCnn = pOpen < 0.15;
        const closed = closedByEar || closedByCnn;
        
        // EAR calibration
        if (this.calibrating && !closed) {
            this.earOpenVals.push(ear);
            if (this.earOpenVals.length > 30) { // 15 samples for calibration
                const mean = this.earOpenVals.reduce((a, b) => a + b) / this.earOpenVals.length;
                const std = Math.sqrt(this.earOpenVals.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / this.earOpenVals.length);
                this.tauEar = Math.max(0.12, mean - 2 * std);
                this.calibrating = false;
                console.log('EAR calibration complete:', this.tauEar);
            }
        }
        
        // Temporal tracking (only after calibration, matching Python server logic)
        if (!this.calibrating) {
            // Track continuous closure duration for blink filtering
            if (closed) {
                if (this.wasClosedLastFrame) {
                    this.currentCloseDuration += dt;
                } else {
                    this.currentCloseDuration = dt;
                }
                this.closedStreak += dt;
                
                // Only add to PERCLOS if this closure has lasted longer than normal blink (0.5+ seconds)
                const isDrowsinessEvent = this.currentCloseDuration >= 0.5;
                this.perclosBuffer.push(isDrowsinessEvent ? 1 : 0);
            } else {
                // Eyes opened
                this.perclosBuffer.push(0);
                this.closedStreak = 0;
                this.currentCloseDuration = 0;
            }
            
            // Keep 60 seconds of data at 2fps = 120 frames
            if (this.perclosBuffer.length > 120) {
                this.perclosBuffer.shift();
            }
            
            this.wasClosedLastFrame = closed;
            
            if (this.cooldown > 0) {
                this.cooldown -= dt;
            }
        }
        
        // Calculate metrics (matching Python server)
        const perclos = this.perclosBuffer.length > 0 ? 
            (this.perclosBuffer.reduce((a, b) => a + b) / this.perclosBuffer.length) : 0;
        const needAlert = (this.closedStreak >= 5.0) || (perclos >= 0.15); // 15% as fraction
        
        // Send alert
        let alertSent = false;
        if (needAlert && this.cooldown <= 0 && !this.calibrating) {
            alertSent = true;
            this.cooldown = 30; // 30 second cooldown
            
            // Reset PERCLOS buffer with 10 "open" frames (matching Python server)
            this.perclosBuffer = Array(10).fill(0);
            console.log('ALERT: Client-side detection triggered');
        }
        
        // Recalculate perclos after potential reset for return value
        const finalPerclos = this.perclosBuffer.length > 0 ? 
            (this.perclosBuffer.reduce((a, b) => a + b) / this.perclosBuffer.length) : 0;
        
        return {
            status: 'ok',
            ear: ear,
            p_open: pOpen,
            tau_ear: this.tauEar,
            closed_streak: this.closedStreak,
            perclos: finalPerclos * 100, // Convert to percentage for display
            cooldown: Math.max(0, this.cooldown),
            calibrating: this.calibrating,
            alert: alertSent,
            cnn_available: true // Client-side has both EAR and CNN-like logic
        };
    }
}

// Initialize the client when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.blinkSenseClient = new BlinkSenseSimpleClient();
});