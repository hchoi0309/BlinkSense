"""
Simple HTTP-based drowsiness detection for testing
"""
import json
import base64
import numpy as np
import logging
import time
from pathlib import Path
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

# Try to import cv2 and mediapipe with graceful fallback
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    CV2_AVAILABLE = False
    logging.warning(f"OpenCV not available: {e}")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    logging.warning(f"MediaPipe not available: {e}")

# Try to import PyTorch and torchvision with graceful fallback
try:
    import torch
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    logging.warning(f"PyTorch/torchvision not available: {e}")
    # Create dummy torch module
    class torch:
        @staticmethod
        def load(*args, **kwargs): return {}
        @staticmethod
        def no_grad(): 
            class NoGradContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return NoGradContext()
        @staticmethod
        def softmax(x, dim): return x
        @staticmethod
        def zeros(*args, **kwargs): return None
        @staticmethod
        def cat(*args, **kwargs): return None
    
    # Create dummy transforms module
    class transforms:
        class Compose:
            def __init__(self, transforms): pass
            def __call__(self, x): return x
        class ToPILImage:
            def __call__(self, x): return x
        class ToTensor:
            def __call__(self, x): return x
        class Normalize:
            def __init__(self, mean, std): pass
            def __call__(self, x): return x

logger = logging.getLogger(__name__)

# Import the trained model
import sys
from pathlib import Path

# Try to find src path more robustly
def find_src_path():
    current_path = Path(__file__).resolve()
    # Try different parent levels to find src directory
    for i in range(5):  # Check up to 5 levels up
        try:
            parent = current_path.parents[i]
            src_path = parent / "src"
            if src_path.exists():
                return str(src_path)
        except IndexError:
            break
    return None

src_path = find_src_path()
if src_path and src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    if TORCH_AVAILABLE:
        try:
            # Try local app import first
            from .simple_resnet_eye import SimpleResNetEye
        except ImportError:
            # Fallback to src path import
            from models.simple_resnet_eye import SimpleResNetEye
        CNN_AVAILABLE = True
        logger.info(f"Successfully imported CNN model")
    else:
        raise ImportError("PyTorch not available")
except ImportError as e:
    CNN_AVAILABLE = False
    logger.warning(f"CNN model not available: {e}")
    # Create dummy class
    class SimpleResNetEye:
        def __init__(self, *args, **kwargs): pass
        def load_state_dict(self, *args, **kwargs): pass
        def eval(self): return self
        def __call__(self, x): return torch.zeros(1, 2), torch.zeros(1, 64)

# Global detector instance
detector = None

class AdvancedDrowsinessDetector:
    """
    Advanced server-side drowsiness detection with EAR + CNN
    """
    
    def __init__(self, model_path="../../model_registry/eye/0.1.0/weights.pt"):
        # Check if required dependencies are available
        if not CV2_AVAILABLE or not MEDIAPIPE_AVAILABLE:
            logger.error("OpenCV and/or MediaPipe not available - advanced detection disabled")
            self.face_mesh = None
        else:
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
        self.model = None
        self.transform = None
        if CNN_AVAILABLE and TORCH_AVAILABLE:
            self.load_cnn_model(model_path)
        
        # Detection state
        self.fps = 5  # Reduced from 15 to account for CNN processing time
        self.p_open_thresh = 0.15  # CNN threshold
        self.hold_sec = 5.0
        self.perclos_sec = 60
        self.perclos_thresh = 0.15
        self.cooldown_sec = 30.0
        self.use_consensus = False  # OR logic: closed if EITHER CNN OR EAR says closed
        
        # Calibration
        self.calibrating = True
        self.tau_ear = 0.20
        self.ear_open_vals = []
        
        # Temporal tracking
        self.perclos_buf = []
        self.closed_streak = 0.0
        self.cooldown = 0.0
        self.last_tick = time.time()
        
        cnn_status = "with CNN" if self.model else "EAR-only"
        logger.info(f"AdvancedDrowsinessDetector initialized {cnn_status}")
    
    def load_cnn_model(self, model_path):
        """Load the trained PyTorch model"""
        try:
            # Try relative path from views_simple.py
            full_path = Path(__file__).resolve().parent / model_path
            if not full_path.exists():
                # Try absolute path from project root
                full_path = Path(__file__).resolve().parents[3] / model_path.lstrip('../')
            
            if not full_path.exists():
                logger.error(f"Model not found at {full_path}")
                return
                
            self.model = SimpleResNetEye(emb_dim=64, num_classes=2)
            state = torch.load(full_path, map_location="cpu")
            
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            
            self.model.load_state_dict(state)
            self.model.eval()
            
            # Setup preprocessing (same as training)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # -> [-1,1]
            ])
            
            logger.info(f"CNN model loaded from {full_path}")
            
        except Exception as e:
            logger.error(f"Failed to load CNN model: {e}")
            self.model = None
            self.transform = None
    
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
    
    def predict_eye_state(self, left_crop, right_crop):
        """Predict if eyes are open/closed using CNN"""
        if not self.model or not self.transform or left_crop is None or right_crop is None:
            return None
            
        try:
            # Preprocess both eyes
            left_tensor = self.transform(left_crop).unsqueeze(0)  # Add batch dimension
            right_tensor = self.transform(right_crop).unsqueeze(0)
            
            # Combine into batch
            batch = torch.cat([left_tensor, right_tensor], dim=0)  # [2, 1, 64, 64]
            
            # Inference
            with torch.no_grad():
                logits, _ = self.model(batch)
                probs = torch.softmax(logits, dim=1)
                
                # Get probabilities for "open" class (class 1 based on dataset labels)
                left_p_open = probs[0, 1].item()
                right_p_open = probs[1, 1].item()
                
                # Use the better-performing eye (max)
                p_open = max(left_p_open, right_p_open)
                
            return p_open
            
        except Exception as e:
            logger.error(f"CNN prediction failed: {e}")
            return None
    
    def update_temporal_state(self, ear, p_open=None):
        """Update temporal detection state and return alert status"""
        now = time.time()
        dt = now - self.last_tick
        self.last_tick = now
        
        # Determine closed state using both EAR and CNN
        closed_by_ear = (ear < self.tau_ear) if ear is not None else None
        closed_by_cnn = (p_open < self.p_open_thresh) if p_open is not None else None
        
        # Combine decisions
        if closed_by_ear is not None and closed_by_cnn is not None:
            # Both signals available
            closed = closed_by_cnn or closed_by_ear if not self.use_consensus else closed_by_cnn and closed_by_ear
        elif closed_by_cnn is not None:
            # CNN only
            closed = closed_by_cnn
        elif closed_by_ear is not None:
            # EAR only
            closed = closed_by_ear
        else:
            return {"status": "no_signal"}
        
        # EAR calibration
        if self.calibrating and ear is not None and not closed:
            self.ear_open_vals.append(ear)
            if len(self.ear_open_vals) > 2 * self.fps:  # Reduced to 2 seconds for very fast testing
                mean_ear = np.mean(self.ear_open_vals)
                std_ear = np.std(self.ear_open_vals)
                self.tau_ear = max(0.12, mean_ear - 2 * std_ear)
                self.calibrating = False
                logger.info(f"EAR calibration complete: tau = {self.tau_ear:.3f}")
        
        # Temporal tracking (only after calibration)
        if not self.calibrating:
            self.perclos_buf.append(1 if closed else 0)
            if len(self.perclos_buf) > int(self.perclos_sec * self.fps):
                self.perclos_buf.pop(0)
            
            if closed:
                self.closed_streak += dt
            else:
                self.closed_streak = 0
                
            if self.cooldown > 0:
                self.cooldown -= dt
        
        # Calculate metrics
        perclos = np.mean(self.perclos_buf) if self.perclos_buf else 0
        need_alert = (self.closed_streak >= self.hold_sec) or (perclos >= self.perclos_thresh)
        
        # Send alert
        alert_sent = False
        if need_alert and self.cooldown <= 0 and not self.calibrating:
            alert_sent = True
            self.cooldown = self.cooldown_sec
            
            # Immediately reset PERCLOS buffer to prevent continuous alerting
            self.perclos_buf = [0] * 10  # Reset with 10 "open" frames
            
            logger.info(f"ALERT: hold={self.closed_streak:.2f}s perclos={perclos*100:.1f}% (PERCLOS cleared)")
            
            # Recalculate perclos for return value after reset
            perclos = np.mean(self.perclos_buf)
        
        return {
            "status": "ok",
            "ear": ear,
            "p_open": p_open,
            "tau_ear": self.tau_ear,
            "closed_streak": self.closed_streak,
            "perclos": perclos * 100,
            "cooldown": max(0, self.cooldown),
            "calibrating": self.calibrating,
            "alert": alert_sent,
            "cnn_available": self.model is not None
        }
    
    def process_frame(self, frame_data):
        """Process a single video frame"""
        try:
            # Check if dependencies are available
            if not CV2_AVAILABLE or not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
                return {"status": "error", "message": "OpenCV or MediaPipe not available"}
            
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
            
            # CNN prediction if available
            p_open = None
            if self.model:
                left_crop = self.crop_eye_region(image, landmarks, self.LEFT_EYE)
                right_crop = self.crop_eye_region(image, landmarks, self.RIGHT_EYE)
                p_open = self.predict_eye_state(left_crop, right_crop)
            
            # Update temporal state with both EAR and CNN
            return self.update_temporal_state(avg_ear, p_open)
            
        except Exception as e:
            logger.error(f"Frame processing failed: {e}")
            return {"status": "error", "message": str(e)}


@csrf_exempt
@require_http_methods(["POST"])
def process_frame(request):
    """HTTP endpoint for frame processing"""
    global detector
    
    # Check if we can initialize the detector
    if detector is None:
        if not CV2_AVAILABLE or not MEDIAPIPE_AVAILABLE:
            return JsonResponse({
                "status": "error", 
                "message": "Server-side processing unavailable. OpenCV or MediaPipe dependencies missing. Please use client-side detection instead."
            })
        detector = AdvancedDrowsinessDetector()
    
    try:
        data = json.loads(request.body)
        frame_data = data.get("frame")
        
        if not frame_data:
            return JsonResponse({"status": "error", "message": "No frame data"})
        
        result = detector.process_frame(frame_data)
        return JsonResponse(result)
        
    except Exception as e:
        logger.error(f"Error in process_frame: {e}")
        return JsonResponse({"status": "error", "message": str(e)})


@csrf_exempt  
@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint for deployment"""
    return JsonResponse({
        "status": "ok",
        "opencv_available": CV2_AVAILABLE,
        "mediapipe_available": MEDIAPIPE_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "cnn_available": CNN_AVAILABLE,
        "advanced_processing": CV2_AVAILABLE and MEDIAPIPE_AVAILABLE
    })