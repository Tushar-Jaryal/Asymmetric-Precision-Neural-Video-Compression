import argparse
import os
import random
import json
import types
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

# --- DISABLE CUSTOM CUDA INFERENCE ---
import src.layers.cuda_inference as cuda_inf
cuda_inf.CUSTOMIZED_CUDA_INFERENCE = False 

from src.models.video_model import DMC
from src.utils.common import get_state_dict
from src.utils.video_reader import PNGReader, YUV420Reader
from src.utils.transforms import rgb2ycbcr, ycbcr420_to_444_np

# ==========================================
# 1. Real Video Dataset (Streaming YUV)
# ==========================================
class DCVCStreamDataset(IterableDataset):
    def __init__(self, config_path, root_path=None):
        self.sequences = []
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        base_root = root_path if root_path else config.get('root_path', '.')
        test_classes = config.get('test_classes', {})

        for class_name, content in test_classes.items():
            if content.get('test', 1) == 0: continue
                
            base_path = os.path.join(base_root, content['base_path'])
            src_type = content['src_type']
            
            for seq_name, seq_info in content['sequences'].items():
                entry = {
                    'path': os.path.join(base_path, seq_name),
                    'width': seq_info['width'],
                    'height': seq_info['height'],
                    'frames': seq_info.get('frames', 100), 
                    'src_type': src_type
                }
                self.sequences.append(entry)
        
        print(f"Dataset Loaded: Found {len(self.sequences)} sequences.")

    def _get_reader(self, seq):
        if seq['src_type'] == 'yuv420':
            return YUV420Reader(seq['path'], seq['width'], seq['height'])
        elif seq['src_type'] == 'png':
            return PNGReader(seq['path'], seq['width'], seq['height'])
        else:
            raise ValueError(f"Unknown src_type: {seq['src_type']}")

    def _pad_tensor(self, x, align=64):
        """Pad tensor [C, H, W] to be divisible by align"""
        h, w = x.shape[-2:]
        pad_h = (align - (h % align)) % align
        pad_w = (align - (w % align)) % align
        if pad_h == 0 and pad_w == 0:
            return x
        # F.pad expects (left, right, top, bottom)
        return F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

    def _to_tensor(self, img_data, src_type):
        """Convert Raw Data -> Float Tensor [3, H, W]"""
        if src_type == 'yuv420':
            y, uv = img_data
            # DCVC util returns [3, H, W] (CHW) already, NOT HWC
            yuv = ycbcr420_to_444_np(y, uv) 
            
            # Direct conversion, NO PERMUTE needed
            tensor = torch.from_numpy(yuv).float() / 255.0
        else:
            # PNG (RGB) [H, W, 3] -> Permute to [3, H, W]
            tensor = torch.from_numpy(img_data).permute(2, 0, 1).float() / 255.0
            tensor = rgb2ycbcr(tensor.unsqueeze(0)).squeeze(0)
            
        return self._pad_tensor(tensor)

    def __iter__(self):
        random.shuffle(self.sequences)
        
        for seq in self.sequences:
            try:
                reader = self._get_reader(seq)
            except Exception as e:
                print(f"Error opening {seq['path']}: {e}")
                continue
                
            try:
                # Read Ref Frame
                ref_data = reader.read_one_frame()
                if ref_data is None: 
                    reader.close()
                    continue
                    
                ref_tensor = self._to_tensor(ref_data, seq['src_type'])
                
                # Limit frames per video to avoid overfitting one sequence
                max_frames = min(seq['frames'], 20) 
                
                for _ in range(max_frames - 1):
                    curr_data = reader.read_one_frame()
                    if curr_data is None: break
                    
                    curr_tensor = self._to_tensor(curr_data, seq['src_type'])
                    
                    yield ref_tensor, curr_tensor
                    ref_tensor = curr_tensor
                    
                reader.close()
            except Exception as e:
                print(f"Error reading {seq['path']}: {e}")
                if 'reader' in locals(): reader.close()
                continue

# ==========================================
# 2. Mock Injection
# ==========================================
class MockEntropyCoder(nn.Module):
    def __init__(self): super().__init__()
    def reset(self): pass
    def flush(self): pass
    def get_encoded_stream(self): return b''
    def set_stream(self, stream): pass
    def set_use_two_entropy_coders(self, use): pass

class MockBitEstimator(nn.Module):
    def __init__(self): super().__init__()
    def encode_z(self, *args, **kwargs): pass
    def decode_z(self, *args, **kwargs): pass
    def get_z(self, *args, **kwargs): return torch.zeros(1)

def inject_mocks(model):
    model.entropy_coder = MockEntropyCoder()
    model.bit_estimator_z = MockBitEstimator()
    if hasattr(model, 'gaussian_encoder'):
        def dummy_encode_y(self, *args, **kwargs): pass
        model.gaussian_encoder.encode_y = types.MethodType(dummy_encode_y, model.gaussian_encoder)
    return model

# ==========================================
# 3. Training Setup
# ==========================================
def configure_optimizers(model, base_lr=1e-5):
    decoder_params = []
    step_size_params = []
    trainable_modules = [model.decoder, model.recon_generation_net]
    
    for module in trainable_modules:
        for name, param in module.named_parameters():
            if not param.requires_grad: continue
            if 's_w' in name or 's_a' in name:
                step_size_params.append(param)
            else:
                decoder_params.append(param)

    optimizer = optim.AdamW([
        {'params': decoder_params, 'lr': base_lr, 'weight_decay': 1e-4},
        {'params': step_size_params, 'lr': 1e-6, 'weight_decay': 0.0}
    ])
    return optimizer

def generate_reconstruction(model, feature, qp):
    q_recon = model.q_recon[qp:qp+1, :, :, :]
    return model.recon_generation_net(feature, q_recon)

def train_epoch(student, teacher, dataloader, optimizer, epoch, device):
    student.train()
    teacher.eval()
    avg_loss = 0
    steps = 0
    qps = [22, 27, 32, 37] 
    
    for i, batch in enumerate(dataloader):
        ref_frame = batch[0].to(device)
        curr_frame = batch[1].to(device)
        
        qp = random.choice(qps)
        optimizer.zero_grad()
        
        # --- Teacher ---
        with torch.no_grad():
            teacher.clear_dpb()
            teacher.add_ref_frame(feature=None, frame=ref_frame)
            teacher.compress(curr_frame, qp)
            teacher_feat = teacher.dpb[0].feature
            target_recon = generate_reconstruction(teacher, teacher_feat, qp)

        # --- Student ---
        student.clear_dpb()
        student.add_ref_frame(feature=None, frame=ref_frame)
        student.compress(curr_frame, qp)
        student_feat = student.dpb[0].feature
        pred_recon = generate_reconstruction(student, student_feat, qp)
        
        loss = nn.MSELoss()(pred_recon, target_recon)
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
        steps += 1
        
        if i % 50 == 0:
            print(f"Epoch {epoch+1} | Step {i} | MSE Loss: {loss.item():.6f}")

    if steps == 0: return 0.0
    return avg_loss / steps

# ==========================================
# 4. Main
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Video FP32 Checkpoint")
    parser.add_argument('--config', type=str, required=True, help="Dataset Config JSON")
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. Load Models
    print("Loading Models...")
    teacher = DMC(use_lsq=False).to(device)
    state_dict = get_state_dict(args.model_path)
    teacher.load_state_dict(state_dict, strict=False)
    teacher = inject_mocks(teacher)
    for p in teacher.parameters(): p.requires_grad = False
    
    student = DMC(use_lsq=True).to(device)
    student.load_state_dict(state_dict, strict=False)
    student = inject_mocks(student)
    for p in student.parameters(): p.requires_grad = False
    for p in student.decoder.parameters(): p.requires_grad = True
    for p in student.recon_generation_net.parameters(): p.requires_grad = True
    
    optimizer = configure_optimizers(student)
    
    # 2. Dataset
    print(f"Loading Real Dataset from {args.config}...")
    dataset = DCVCStreamDataset(args.config, root_path=args.dataset_root)
    dataloader = DataLoader(dataset, batch_size=1)
    
    # 3. Train
    for epoch in range(args.epochs):
        print(f"\n--- Starting Epoch {epoch+1} ---")
        avg_loss = train_epoch(student, teacher, dataloader, optimizer, epoch, device)
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.6f}")
        
        save_path = os.path.join(args.save_dir, f"dcvc_lsq_real_epoch{epoch+1}.pth")
        torch.save(student.state_dict(), save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()