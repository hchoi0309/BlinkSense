"""
WebSocket consumer for real-time drowsiness detection
"""
import json
import base64
import numpy as np
import cv2
import torch
import logging
import time
from pathlib import Path
from channels.generic.websocket import AsyncWebsocketConsumer
import mediapipe as mp

logger = logging.getLogger(__name__)

# Handle torchvision import gracefully
try:
    from torchvision import transforms
    HAS_TORCHVISION = True
except ImportError as e:
    logger.warning(f"Torchvision not available: {e}")
    HAS_TORCHVISION = False
    # Define minimal transforms fallback
    class Transforms:
        @staticmethod
        def ToPILImage():
            return lambda x: x
        @staticmethod  
        def ToTensor():
            return lambda x: torch.from_numpy(x).float().unsqueeze(0) / 255.0
        @staticmethod
        def Normalize(mean, std):
            return lambda x: (x - mean[0]) / std[0]
        @staticmethod
        def Compose(transforms_list):
            def compose(x):
                for t in transforms_list:
                    x = t(x)
                return x
            return compose
    transforms = Transforms()

# Import the model - add parent directory to path
import sys
src_path = str(Path(__file__).resolve().parents[2] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from models.simple_resnet_eye import SimpleResNetEye
    logger.info("Successfully imported real SimpleResNetEye model")
except ImportError as e:
    logger.error(f"Failed to import SimpleResNetEye: {e}")
    # Fallback dummy model
    class SimpleResNetEye:
        def __init__(self, *args, **kwargs):
            logger.warning("Using dummy model - CNN predictions will not work!")
            pass
        def load_state_dict(self, *args, **kwargs):
            pass
        def eval(self):
            return self
        def __call__(self, x):
            return torch.zeros(1, 2), torch.zeros(1, 64)

class DrowsinessDetector:
    """
    Server-side drowsiness detection using MediaPipe + CNN + temporal logic
    """
    
    def __init__(self, model_path=None):
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices (same as JS version)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        # Load CNN model
        self.model = SimpleResNetEye(emb_dim=64, num_classes=2)
        if model_path is None:
            # Default path from project root
            model_path = Path(__file__).resolve().parents[2] / "model_registry" / "eye" / "0.1.0" / "weights.pt"
        self.load_model(model_path)
        
        # Image preprocessing (same as training)
        if HAS_TORCHVISION:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # -> [-1,1]
            ])
        else:
            # Fallback preprocessing without torchvision
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        
        # Detection state
        self.fps = 15
        self.p_open_thresh = 0.3  # More conservative - only count as closed if CNN is quite confident
        self.hold_sec = 5.0
        self.perclos_sec = 30  # Reduced from 60s to 30s for faster recovery
        self.perclos_thresh = 0.20  # Raise PERCLOS alert threshold from 15% to 20%
        self.cooldown_sec = 10.0
        self.min_blink_duration = 2.0  # Only count closures lasting 2+ seconds for PERCLOS
        
        # Calibration
        self.calibrating = True
        self.tau_ear = 0.20
        self.ear_open_vals = []
        
        # Temporal tracking
        self.perclos_buf = []
        self.closed_streak = 0.0
        self.cooldown = 0.0
        self.last_tick = time.time()
        
        # Track sustained closures for PERCLOS
        self.current_closure_duration = 0.0
        self.sustained_closures_buffer = []  # Track periods that qualify for PERCLOS
        
    def load_model(self, model_path):
        """Load the trained PyTorch model"""
        try:
            # Convert to Path object if it's a string
            if isinstance(model_path, str):
                model_path = Path(model_path)
            
            if not model_path.exists():
                logger.error(f"Model not found at {model_path}")
                return
                
            state = torch.load(model_path, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            
            self.model.load_state_dict(state)
            self.model.eval()
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def ear_from_landmarks(self, landmarks, indices):
        """Calculate Eye Aspect Ratio from face landmarks"""
        points = [landmarks[i] for i in indices]
        
        # Vertical distances
        vert1 = np.linalg.norm(np.array([points[1].x, points[1].y]) - np.array([points[5].x, points[5].y]))
        vert2 = np.linalg.norm(np.array([points[2].x, points[2].y]) - np.array([points[4].x, points[4].y]))
        
        # Horizontal distance  
        hor = np.linalg.norm(np.array([points[0].x, points[0].y]) - np.array([points[3].x, points[3].y]))
        
        return (vert1 + vert2) / (2.0 * hor + 1e-6)
    
    def crop_eye_region(self, image, landmarks, indices, size=64, scale=2.0):
        """Crop eye region from image using face landmarks"""
        points = [(int(lm.x * image.shape[1]), int(lm.y * image.shape[0])) 
                  for lm in [landmarks[i] for i in indices]]
        
        # Calculate bounding box
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Expand bounding box
        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        s = max(x_max - x_min, y_max - y_min) * scale
        
        x1, y1 = int(cx - s//2), int(cy - s//2)
        x2, y2 = int(cx + s//2), int(cy + s//2)
        
        # Ensure bounds are within image
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Crop and resize to 64x64 grayscale
        crop = image[y1:y2, x1:x2]
        if len(crop.shape) == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        
        return crop
    
    def predict_eye_state(self, eye_crop):
        """Predict if eye is open/closed using CNN"""
        try:
            # Preprocess - handle numpy array input
            if not HAS_TORCHVISION:
                # Manual preprocessing without torchvision
                if isinstance(eye_crop, np.ndarray):
                    # Convert to tensor: HWC -> CHW and normalize
                    tensor = torch.from_numpy(eye_crop).float()
                    if len(tensor.shape) == 2:  # Grayscale
                        tensor = tensor.unsqueeze(0)  # Add channel dim
                    elif len(tensor.shape) == 3:  # HWC -> CHW
                        tensor = tensor.permute(2, 0, 1)
                    tensor = tensor / 255.0  # [0,1]
                    tensor = (tensor - 0.5) / 0.5  # [-1,1]
                    tensor = tensor.unsqueeze(0)  # Add batch dimension
                else:
                    tensor = self.transform(eye_crop).unsqueeze(0)
            else:
                tensor = self.transform(eye_crop).unsqueeze(0)  # Add batch dimension
            
            # Inference
            with torch.no_grad():
                logits, _ = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                p_open = probs[0, 1].item()  # Class 1 = OPEN, Class 0 = CLOSED
                
            return p_open
            
        except Exception as e:
            logger.error(f"CNN prediction failed: {e}")
            return None
    
    def update_temporal_state(self, ear, p_open):
        """Update temporal detection state and return alert status"""
        now = time.time()
        dt = now - self.last_tick
        self.last_tick = now
        
        # Determine closed state
        closed_by_ear = (ear < self.tau_ear) if ear is not None else None
        closed_by_prob = (p_open < self.p_open_thresh) if p_open is not None else None
        
        if closed_by_ear is not None and closed_by_prob is not None:
            closed = closed_by_prob or closed_by_ear  # OR logic
        elif closed_by_prob is not None:
            closed = closed_by_prob
        elif closed_by_ear is not None:
            closed = closed_by_ear
        else:
            return {"status": "no_signal"}
        
        # EAR calibration
        if self.calibrating and ear is not None and not closed_by_ear:
            self.ear_open_vals.append(ear)
            if len(self.ear_open_vals) > 45 * self.fps:
                mean_ear = np.mean(self.ear_open_vals)
                std_ear = np.std(self.ear_open_vals)
                self.tau_ear = max(0.12, mean_ear - 2 * std_ear)
                self.calibrating = False
                logger.info(f"EAR calibration complete: tau = {self.tau_ear:.3f}")
        
        # Temporal tracking (only after calibration)
        if not self.calibrating:
            # Track closed streak (for immediate alerts)
            if closed:
                self.closed_streak += dt
                self.current_closure_duration += dt
            else:
                # End of closure - check if it was sustained enough for PERCLOS
                if self.current_closure_duration >= self.min_blink_duration:
                    # Add this sustained closure period to PERCLOS tracking
                    num_frames = int(self.current_closure_duration * self.fps)
                    for _ in range(num_frames):
                        self.sustained_closures_buffer.append(1)
                
                self.closed_streak = 0
                self.current_closure_duration = 0.0
            
            # Also add open frames to maintain the time window
            if not closed:
                self.sustained_closures_buffer.append(0)
            
            # Maintain window size for PERCLOS calculation
            window_size = int(self.perclos_sec * self.fps)
            if len(self.sustained_closures_buffer) > window_size:
                self.sustained_closures_buffer = self.sustained_closures_buffer[-window_size:]
                
            if self.cooldown > 0:
                self.cooldown -= dt
        
        # Calculate metrics
        perclos = np.mean(self.sustained_closures_buffer) if self.sustained_closures_buffer else 0
        need_alert = (self.closed_streak >= self.hold_sec) or (perclos >= self.perclos_thresh)
        
        # Send alert
        alert_sent = False
        if need_alert and self.cooldown <= 0 and not self.calibrating:
            alert_sent = True
            self.cooldown = self.cooldown_sec
            logger.info(f"ALERT: hold={self.closed_streak:.2f}s perclos={perclos*100:.1f}%")
        
        return {
            "status": "ok",
            "ear": ear,
            "p_open": p_open,
            "tau_ear": self.tau_ear,
            "closed_streak": self.closed_streak,
            "perclos": perclos * 100,
            "cooldown": max(0, self.cooldown),
            "calibrating": self.calibrating,
            "alert": alert_sent
        }
    
    def process_frame(self, frame_data):
        """Process a single video frame"""
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(frame_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"status": "invalid_frame"}
            
            # Face detection
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return {"status": "no_face"}
            
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate EAR
            left_ear = self.ear_from_landmarks(landmarks, self.LEFT_EYE)
            right_ear = self.ear_from_landmarks(landmarks, self.RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2
            
            # Crop eyes and predict
            left_crop = self.crop_eye_region(image, landmarks, self.LEFT_EYE)
            right_crop = self.crop_eye_region(image, landmarks, self.RIGHT_EYE)
            
            p_open = None
            if left_crop is not None and right_crop is not None:
                left_p = self.predict_eye_state(left_crop)
                right_p = self.predict_eye_state(right_crop)
                if left_p is not None and right_p is not None:
                    p_open = max(left_p, right_p)  # Use best eye
            
            # Update temporal state
            return self.update_temporal_state(avg_ear, p_open)
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return {"status": "error", "message": str(e)}


class DrowsinessConsumer(AsyncWebsocketConsumer):
    """WebSocket consumer for drowsiness detection"""
    
    async def connect(self):
        await self.accept()
        self.detector = DrowsinessDetector()
        logger.info("WebSocket connected")
    
    async def disconnect(self, close_code):
        logger.info(f"WebSocket disconnected: {close_code}")
    
    async def receive(self, text_data):
        """Receive video frame and process"""
        try:
            data = json.loads(text_data)
            
            if data.get("type") == "frame":
                frame_data = data.get("frame")
                if frame_data:
                    result = self.detector.process_frame(frame_data)
                    await self.send(text_data=json.dumps(result))
                    
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.send(text_data=json.dumps({
                "status": "error", 
                "message": str(e)
            }))