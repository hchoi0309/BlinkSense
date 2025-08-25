/**
 * BlinkSense Client - Minimal WebSocket-based drowsiness detection client
 * 
 * This client captures video frames and sends them to the server for processing.
 * All the heavy lifting (MediaPipe, CNN, temporal logic) is done server-side.
 */

class BlinkSenseClient {
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
        this.ws = null;
        this.stream = null;
        this.isRunning = false;
        this.frameInterval = null;
        
        // Settings
        this.frameRate = 15; // FPS to send to server
        this.jpegQuality = 0.7; // Compression quality
        
        this.setupEventListeners();
        this.setupCanvas();
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
            
            // Connect WebSocket
            await this.connectWebSocket();
            
            // Start sending frames
            this.startFrameCapture();
            
            this.isRunning = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.detectionStatus.textContent = 'Starting...';
            
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
    
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/drowsiness/`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus(true);
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                this.handleServerMessage(JSON.parse(event.data));
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
                reject(new Error('WebSocket connection failed'));
            };
            
            // Timeout after 5 seconds
            setTimeout(() => {
                if (this.ws.readyState !== WebSocket.OPEN) {
                    reject(new Error('WebSocket connection timeout'));
                }
            }, 5000);
        });
    }
    
    startFrameCapture() {
        const interval = 1000 / this.frameRate; // Convert FPS to milliseconds
        
        this.frameInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.captureAndSendFrame();
            }
        }, interval);
    }
    
    captureAndSendFrame() {
        try {
            // Draw current video frame to canvas
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Convert to base64 JPEG
            const dataURL = this.canvas.toDataURL('image/jpeg', this.jpegQuality);
            const base64Data = dataURL.split(',')[1]; // Remove data:image/jpeg;base64, prefix
            
            // Send to server
            this.ws.send(JSON.stringify({
                type: 'frame',
                frame: base64Data
            }));
            
        } catch (error) {
            console.error('Frame capture failed:', error);
        }
    }
    
    handleServerMessage(data) {
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
            this.detectionStatus.textContent = 'Active Detection';
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
    
    updateConnectionStatus(connected) {
        if (connected) {
            this.connectionStatus.className = 'connection-status connected';
            this.connectionText.textContent = 'Connected';
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
        
        // Close WebSocket
        if (this.ws) {
            this.ws.close();
            this.ws = null;
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
    }
}

// Initialize the client when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.blinkSenseClient = new BlinkSenseClient();
});