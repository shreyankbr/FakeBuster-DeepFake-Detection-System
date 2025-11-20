from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
import timm
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import io
import time
import base64
from flask_cors import CORS
from pathlib import Path

# --- 1. Configuration ---
NUM_FRAMES_TEST = 18 
THRESHOLD_FINAL = 0.45 
MODEL_12F_PATH = "12f/5.pth" 
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
IMG_SIZE = 224

# --- 2. Model Definition and Helper Functions ---
class DualBranchEfficientNet(nn.Module):
    def __init__(self, embed_dim=512, num_classes=1, dropout=0.3):
        super().__init__()
        try:
            self.rgb_backbone = timm.create_model("tf_efficientnet_b4.ns_jft_in1k", pretrained=False, num_classes=0, global_pool="avg")
            self.fft_backbone = timm.create_model("tf_efficientnet_b0.ns_jft_in1k", pretrained=False, in_chans=1, num_classes=0, global_pool="avg")
        except Exception:
             print("Warning: Using potentially deprecated timm model names ('_ns'). Update timm if possible.")
             self.rgb_backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=False, num_classes=0, global_pool="avg")
             self.fft_backbone = timm.create_model("tf_efficientnet_b0_ns", pretrained=False, in_chans=1, num_classes=0, global_pool="avg")

        rgb_dim = self.rgb_backbone.num_features
        self.rgb_proj = nn.Sequential(
            nn.Linear(rgb_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        fft_dim = self.fft_backbone.num_features
        self.fft_proj = nn.Sequential(
            nn.Linear(fft_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, rgb, fft):
        B, F, _, H, W = rgb.shape
        rgb_flat = rgb.view(B * F, 3, H, W)
        rgb_feats = self.rgb_backbone(rgb_flat)
        rgb_feats = self.rgb_proj(rgb_feats)
        rgb_feats = rgb_feats.view(B, F, -1).permute(0, 2, 1)
        rgb_pooled = self.temporal_pool(rgb_feats).squeeze(-1)

        fft_flat = fft.view(B * F, 1, H, W)
        fft_feats = self.fft_backbone(fft_flat)
        fft_feats = self.fft_proj(fft_feats)
        fft_feats = fft_feats.view(B, F, -1).permute(0, 2, 1)
        fft_pooled = self.temporal_pool(fft_feats).squeeze(-1)

        fused = torch.cat([rgb_pooled, fft_pooled], dim=1)
        out = self.classifier(fused)
        return out

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def to_fft_tensor(img_tensor: torch.Tensor) -> torch.Tensor:
    gray = img_tensor.mean(dim=0, keepdim=True)
    fft = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft)
    magnitude = torch.abs(fft_shift)
    log_mag = torch.log1p(magnitude)
    min_val, max_val = log_mag.min(), log_mag.max()
    if (max_val - min_val) > 1e-8:
        log_mag = (log_mag - min_val) / (max_val - min_val)
    else:
        log_mag = torch.zeros_like(log_mag)
    return log_mag

def load_model_from_path(model_path, device):
    model = DualBranchEfficientNet(embed_dim=512, num_classes=1, dropout=0.3).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Champion Model loaded: {model_path}")
        if "val_auc" in checkpoint:
            print(f"   Validation AUC: {checkpoint['val_auc']:.4f}")
        return model
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Model file not found at {model_path}. Exiting.")
        exit()
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load model {model_path}: {e}")
        exit()

def extract_frames_from_video(video_path, num_frames):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return frames
    indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    # Pad if necessary
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])
    if not frames:
        frames = [Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')] * num_frames
    return frames[:num_frames]

def load_image_frames(image_path, num_frames):
    try:
        img = Image.open(image_path).convert("RGB")
        return [img] * num_frames
    except Exception as e:
        black_frame = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
        return [black_frame] * num_frames

def get_probability(content_path, model, device, num_frames):
    content_path = Path(content_path)
    frames = []
    if content_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        frames = extract_frames_from_video(content_path, num_frames)
    elif content_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        frames = load_image_frames(content_path, num_frames)
    else:
        return None
    min_required_frames = 8
    if not frames or len(frames) < min_required_frames:
        if frames:
            while len(frames) < num_frames:
                frames.append(frames[-1] if frames else Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black'))
        else:
            frames = [Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')] * num_frames
    rgb_imgs, fft_imgs = [], []
    for i in range(num_frames):
        current_frame = frames[i] if i < len(frames) else (frames[-1] if frames else Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black'))
        try:
            rgb = transform(current_frame)
            rgb_imgs.append(rgb)
            fft_map = to_fft_tensor(rgb)
            fft_imgs.append(fft_map)
        except Exception as e:
            black_frame_rgb = transform(Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black'))
            rgb_imgs.append(black_frame_rgb)
            fft_imgs.append(to_fft_tensor(black_frame_rgb))
    if not rgb_imgs:
        black_frame_rgb = transform(Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black'))
        rgb_imgs = [black_frame_rgb] * num_frames
        fft_imgs = [to_fft_tensor(black_frame_rgb)] * num_frames
    try:
        rgb_batch = torch.stack(rgb_imgs, dim=0).unsqueeze(0).to(device)
        fft_batch = torch.stack(fft_imgs, dim=0).unsqueeze(0).to(device)
    except RuntimeError as e:
        print(f"Stack Error: {e}")
        return None
    with torch.no_grad():
        try:
            output = model(rgb_batch, fft_batch)
            probability = torch.sigmoid(output).item()
        except Exception as e:
            print(f"Inference Error: {e}")
            return None
    return probability

def get_champion_prediction(model, device, frames_list):
    if not frames_list or len(frames_list) != NUM_FRAMES_TEST:
        return None
    rgb_tensors = []
    fft_tensors = []
    for frame in frames_list:
        try:
            rgb = transform(frame)
            rgb_tensors.append(rgb)
            fft_tensors.append(to_fft_tensor(rgb))
        except Exception as e:
            black_frame_rgb = transform(Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black'))
            rgb_tensors.append(black_frame_rgb)
            fft_tensors.append(to_fft_tensor(black_frame_rgb))
    try:
        rgb_batch = torch.stack(rgb_tensors).unsqueeze(0).to(device)
        fft_batch = torch.stack(fft_tensors).unsqueeze(0).to(device)
    except RuntimeError as e:
        print(f"Stack Error: {e}")
        return None
    probability = None
    with torch.no_grad():
        try:
            logits = model(rgb_batch, fft_batch)
            probability = torch.sigmoid(logits).item()
        except Exception as e:
            print(f"Inference Error: {e}")
    return probability

# --- 3. Grad-CAM Implementation ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        handle_forward = self.target_layer.register_forward_hook(forward_hook)
        handle_backward = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([handle_forward, handle_backward])

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def generate_cam(self, rgb_input_tensor, fft_input_tensor):
        self.model.zero_grad()
        if rgb_input_tensor.dim() == 4:
            rgb_input_tensor = rgb_input_tensor.unsqueeze(0)
        if fft_input_tensor.dim() == 4:
            fft_input_tensor = fft_input_tensor.unsqueeze(0)
        output = self.model(rgb_input_tensor, fft_input_tensor)
        target_output = output
        target_output.backward(gradient=torch.ones_like(target_output), retain_graph=True)
        if self.gradients is None or self.activations is None:
            print("CAM Error: Hooks failed.")
            return np.zeros((IMG_SIZE, IMG_SIZE))
        grads = self.gradients
        acts = self.activations
        B = rgb_input_tensor.shape[0]
        F = rgb_input_tensor.shape[1]
        pooled_gradients = torch.mean(grads, dim=[2, 3])
        acts_reshaped = acts.view(B, F, acts.size(1), acts.size(2), acts.size(3))
        pooled_gradients_reshaped = pooled_gradients.view(B, F, -1)
        first_frame_activations = acts_reshaped[0, 0, :, :, :]
        first_frame_pooled_grads = pooled_gradients_reshaped[0, 0, :]
        for i in range(first_frame_activations.size(0)):
            first_frame_activations[i, :, :] *= first_frame_pooled_grads[i]
        heatmap = torch.mean(first_frame_activations, dim=0)
        heatmap = heatmap.squeeze()
        heatmap = torch.relu(heatmap)
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        return heatmap.cpu().detach().numpy()

def apply_gradcam_and_encode(cam_model, original_image_pil, device):
    try:
        if not isinstance(original_image_pil, Image.Image):
            raise TypeError("Input must be a PIL Image.")
        img_cv = cv2.cvtColor(np.array(original_image_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        rgb_list = []
        fft_list = []
        for _ in range(NUM_FRAMES_TEST):
            rgb_t = transform(original_image_pil)
            rgb_list.append(rgb_t)
            fft_list.append(to_fft_tensor(rgb_t))
        rgb_tensor_cam = torch.stack(rgb_list).to(device)
        fft_tensor_cam = torch.stack(fft_list).to(device)
        rgb_tensor_cam = rgb_tensor_cam.unsqueeze(0)
        fft_tensor_cam = fft_tensor_cam.unsqueeze(0)
        try:
            target_layer = cam_model.rgb_backbone.conv_head
        except AttributeError:
            print("CAM Error: Target layer not found.")
            return None
        grad_cam_gen = GradCAM(cam_model, target_layer)
        heatmap = grad_cam_gen.generate_cam(rgb_tensor_cam, fft_tensor_cam)
        grad_cam_gen.remove_hooks()
        if heatmap is None or heatmap.size == 0:
            print("CAM Warning: Heatmap generation failed.")
            return None
        heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap_colored * 0.4 + img_cv * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        pil_result = Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        buff = io.BytesIO()
        pil_result.save(buff, format="JPEG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Grad-CAM Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- 4. Flask App Setup ---
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- 5. Load Model ---
print("🚀 Starting server and loading Champion Model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
champion_model = load_model_from_path(MODEL_12F_PATH, device)
print("✅ Champion model loaded successfully!")
print(f"🎯 Configuration: {NUM_FRAMES_TEST} frames @ threshold {THRESHOLD_FINAL}")

# --- 6. API Routes ---
@app.route('/api/analyze/image', methods=['POST'])
def analyze_image_route():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file part found'}), 400
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pil_image = None
    try:
        file.save(image_path)
        pil_image = Image.open(image_path).convert("RGB")
        frames = [pil_image] * NUM_FRAMES_TEST
        
        probability = get_champion_prediction(champion_model, device, frames)
        if probability is None:
            raise RuntimeError("Inference failed.")

        raw_probability_fake = probability  
        prediction = "FAKE" if raw_probability_fake > THRESHOLD_FINAL else "REAL"
        confidence_in_prediction = max(raw_probability_fake, 1 - raw_probability_fake)

        heatmap_base64 = apply_gradcam_and_encode(champion_model, pil_image, device)

        result = {
            'prediction': prediction,
            'isReal': prediction == "REAL",
            'confidence': confidence_in_prediction,
            'probability': raw_probability_fake,
            'heatmap_base64': heatmap_base64,
            'model': 'champion_12f_5'
        }

        return jsonify(result)
    except Exception as e:
        print(f"Error in /api/analyze/image: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e_clean:
                print(f"Warning: could not remove {image_path}: {e_clean}")

@app.route('/api/analyze/video', methods=['POST'])
def analyze_video_route():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file part found'}), 400
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    first_frame_for_cam = None
    try:
        file.save(video_path)
        frames = extract_frames_from_video(video_path, NUM_FRAMES_TEST)
        if not frames or len(frames) < NUM_FRAMES_TEST:
            raise ValueError(f"Could not extract {NUM_FRAMES_TEST} frames")
        if frames:
            first_frame_for_cam = frames[0]
            
        probability = get_champion_prediction(champion_model, device, frames)
        if probability is None:
            raise RuntimeError("Inference failed.")

        raw_probability_fake = probability
        prediction = "FAKE" if raw_probability_fake > THRESHOLD_FINAL else "REAL"
        confidence_in_prediction = max(raw_probability_fake, 1 - raw_probability_fake)

        heatmap_base64 = None
        if first_frame_for_cam:
            heatmap_base64 = apply_gradcam_and_encode(champion_model, first_frame_for_cam, device)

        result = {
            'prediction': prediction,
            'isReal': prediction == "REAL",
            'confidence': confidence_in_prediction, 
            'probability': raw_probability_fake,
            'heatmap_base64': heatmap_base64,
            'frame_count': len(frames),
            'model': 'champion_12f_5'
        }

        return jsonify(result)
    except Exception as e:
        print(f"Error in /api/analyze/video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e_clean:
                print(f"Warning: could not remove {video_path}: {e_clean}")

@app.route('/api/analyze/batch', methods=['POST'])
def analyze_batch_route():
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'No files found'}), 400
    files = files[:NUM_FRAMES_TEST]
    frames_pil = []
    saved_files = []
    first_pil_frame = None
    try:
        for i, file in enumerate(files):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)
            saved_files.append(image_path)
            try:
                img = Image.open(image_path).convert("RGB")
                frames_pil.append(img)
            except Exception as e:
                print(f"Warning: Skipping {filename}: {e}")
                continue
        if not frames_pil:
            return jsonify({'error': 'No valid images processed'}), 500
        # Pad if needed
        while len(frames_pil) < NUM_FRAMES_TEST and len(frames_pil) > 0:
            frames_pil.append(frames_pil[-1])
        if not frames_pil:
            return jsonify({'error': 'Could not process images'}), 500
        if frames_pil:
            first_pil_frame = frames_pil[0]

        probability = get_champion_prediction(champion_model, device, frames_pil)
        if probability is None:
            raise RuntimeError("Inference failed.")

        raw_probability_fake = probability
        prediction = "FAKE" if raw_probability_fake > THRESHOLD_FINAL else "REAL"
        confidence_in_prediction = max(raw_probability_fake, 1 - raw_probability_fake)

        heatmap_base64 = None
        if first_pil_frame:
            heatmap_base64 = apply_gradcam_and_encode(champion_model, first_pil_frame, device)

        result = {
            'prediction': prediction,
            'isReal': prediction == "REAL",
            'confidence': confidence_in_prediction,
            'probability': raw_probability_fake,
            'heatmap_base64': heatmap_base64,
            'frame_count': len(frames_pil),
            'type': 'video_frames_batch',
            'model': 'champion_12f_5'
        }

        return jsonify(result)
    except Exception as e:
        print(f"Error in /api/analyze/batch: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        for f_path in saved_files:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except Exception as e_clean:
                    print(f"Warning: could not remove {f_path}: {e_clean}")

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model': 'champion_12f_5',
        'frames': NUM_FRAMES_TEST,
        'threshold': THRESHOLD_FINAL,
        'timestamp': time.time()
    })

# --- 7. HTML Serving Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/history')
def history():
    return render_template('history.html')

# --- 8. Run App ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)