#!/usr/bin/env python3
"""
Verify the eye state CNN model by testing on sample images from the dataset.
This helps debug issues with low p_open values in the web interface.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from torchvision import transforms

# Import the model
from src.models.simple_resnet_eye import SimpleResNetEye

def softmax(logits):
    """Manual softmax for debugging"""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def preprocess_image(img_path, eye_size=64):
    """Preprocess image exactly like the training pipeline"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    
    # Resize to 64x64
    if img.shape != (eye_size, eye_size):
        img = cv2.resize(img, (eye_size, eye_size), interpolation=cv2.INTER_AREA)
    
    # Apply the same transforms as validation (no augmentation)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # -> [-1,1]
    ])
    
    return transform(img).unsqueeze(0)  # Add batch dimension

def main():
    print("=== BlinkSense Model Verification ===\n")
    
    # Load model
    model_path = Path("model_registry/eye/0.1.0/weights.pt")
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    print(f"Loading model from: {model_path}")
    model = SimpleResNetEye(emb_dim=64, num_classes=2).eval()
    
    # Load weights
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    print("✓ Model loaded successfully\n")
    
    # Load dataset info
    csv_path = Path("data/mrl/meta/mrl_index.csv")
    if not csv_path.exists():
        print(f"ERROR: Dataset CSV not found at {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Dataset contains {len(df)} images from {df['subject_id'].nunique()} subjects\n")
    
    # Test on some sample images
    print("Testing on sample images:")
    print("=" * 50)
    
    # Get a few open and closed eye samples
    open_samples = df[df['eye_state'] == 1].sample(3, random_state=42)
    closed_samples = df[df['eye_state'] == 0].sample(3, random_state=42)
    
    all_samples = pd.concat([open_samples, closed_samples])
    
    for _, row in all_samples.iterrows():
        img_path = Path("data/mrl") / row["img_relpath"]
        expected_label = row["eye_state"]  # 0=closed, 1=open
        expected_state = "OPEN" if expected_label == 1 else "CLOSED"
        
        try:
            # Preprocess image
            x = preprocess_image(img_path)
            
            # Run inference
            with torch.no_grad():
                logits, emb = model(x)
                logits_np = logits.cpu().numpy()[0]  # Remove batch dimension
                
                # Calculate probabilities
                probs = softmax(logits_np)
                p_closed = probs[0]
                p_open = probs[1]
                
                predicted_label = np.argmax(probs)
                predicted_state = "OPEN" if predicted_label == 1 else "CLOSED"
                
                # Check if prediction matches expected
                correct = "✓" if predicted_label == expected_label else "✗"
                
                print(f"Subject {row['subject_id']:>2} | Expected: {expected_state:<6} | "
                      f"Predicted: {predicted_state:<6} | "
                      f"p_closed: {p_closed:.3f} | p_open: {p_open:.3f} | {correct}")
                
        except Exception as e:
            print(f"ERROR processing {img_path}: {e}")
    
    print("\n" + "=" * 50)
    print("Key observations:")
    print("- p_open should be HIGH (>0.5) for open eyes")
    print("- p_open should be LOW (<0.5) for closed eyes")
    print("- If p_open is always low, check:")
    print("  1. Image preprocessing (normalization, resizing)")
    print("  2. Model input format (NCHW vs NHWC)")
    print("  3. Class label mapping (0=closed, 1=open)")
    print("  4. ONNX export correctness")

if __name__ == "__main__":
    main()