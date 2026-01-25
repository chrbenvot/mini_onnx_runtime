# THis was also ran on kaggle and only here for doc
import onnxruntime as ort
import numpy as np
import cv2
import sys

# --- CONFIGURATION ---
MODEL_PATH = "/kaggle/input/onxx-test/onnx/default/1/tiny_yolo.onnx"
IMAGE_PATH = "/kaggle/input/onnx-test-dog/dog.jpg" 

def preprocess_cplusplus_style(img_path):
    """
    Mimics current C++ 'letterbox' + '0-255' preprocessing EXACTLY.
    """
    # 1. Load Image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not load {img_path}")
        sys.exit(1)
    
    original_h, original_w = img.shape[:2]
    target_w, target_h = 416, 416

    # 2. Letterbox Logic (Same as your C++ voc_eval)
    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    # Create gray canvas
    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    
    # Center paste
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    # 3. BGR to RGB
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # 4. Transpose HWC -> CHW (Channels, Height, Width)
    # C++ does this in the loop. Numpy does it with transpose.
    img_chw = img_rgb.transpose(2, 0, 1)

    # 5. Convert to Float32
    # CRITICAL: Our C++ uses 0-255 range. 
    # If standard YOLO, this should be / 255.0. 
    # But we are testing OUR C++ logic, so we keep 0-255.
    img_float = img_chw.astype(np.float32) 
    
    # Add Batch Dimension [1, 3, 416, 416]
    img_tensor = np.expand_dims(img_float, axis=0)
    
    return img_tensor

def run_inference():
    # 1. Load Engine
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    
    # 2. Prepare Data
    data = preprocess_cplusplus_style(IMAGE_PATH)
    
    # 3. Run ONNX Runtime (The "Ground Truth")
    outputs = session.run(None, {input_name: data})
    raw_output = outputs[0].flatten() # Flatten to 1D array

    # 4. Print Stats for Comparison
    print(f"--- Python Reference Output ---")
    print(f"Input Shape: {data.shape}")
    print(f"Output Count: {len(raw_output)} floats")
    print(f"Min Value:   {raw_output.min():.6f}")
    print(f"Max Value:   {raw_output.max():.6f}")
    print(f"Mean Value:  {raw_output.mean():.6f}")
    print("-" * 30)
    print("First 5 values: ", raw_output[:5])
    print("Center 5 values:", raw_output[len(raw_output)//2 : len(raw_output)//2 + 5])
    print("-" * 30)

    # 5. Check for valid detections (Quick Check)
    # Look for objectness score > -2.0 (sigmoid(-2) approx 0.12)
    # The 5th channel (index 4) in every block is objectness
    grid_size = 13 * 13
    num_anchors = 5
    block_size = 5 + 20 # 25
    
    # This is rough indexing just to see if ANY box is confident
    confident_boxes = 0
    for i in range(len(raw_output)):
        # Every 25th value is objectness? Roughly.
        # Just scanning simple threshold
        if raw_output[i] > 0.0: # Sigmoid(0) = 0.5
            confident_boxes += 1
            
    print(f"Values > 0.0 (Potential Objects): {confident_boxes}")

if __name__ == "__main__":
    run_inference()
