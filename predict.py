import torch
import nibabel as nib
import numpy as np
import os
import glob
from model import UNet3D
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_IMG_DIR = "/hy-tmp/TOF-MRA/test/raw/"
OUTPUT_DIR = "/hy-tmp/TOF-MRA/predictions/"
MODEL_PATH = "my_checkpoint.pth.tar"
PATCH_SIZE = (64, 128, 128) # (D, H, W) in model input terms.
# Wait, model input is (N, C, D, H, W)
# Our dataset returns (C, D, H, W) = (1, 64, 128, 128)

def sliding_window_inference(model, image, patch_size=(64, 128, 128), stride=(32, 64, 64)):
    # image shape: (C, D, H, W)
    C, D, H, W = image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride
    
    # Calculate padded dimensions
    pad_d = max(0, pd - D)
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    
    # Pad image
    # Padding: (W_left, W_right, H_top, H_bottom, D_front, D_back)
    image = torch.nn.functional.pad(image.unsqueeze(0), (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0).squeeze(0)
    
    _, D_pad, H_pad, W_pad = image.shape

    # Output buffers with padded size
    output = torch.zeros((C, D_pad, H_pad, W_pad), device=DEVICE)
    count_map = torch.zeros((C, D_pad, H_pad, W_pad), device=DEVICE)
    
    # Generate patch coordinates
    d_steps = range(0, D_pad - pd + 1, sd)
    h_steps = range(0, H_pad - ph + 1, sh)
    w_steps = range(0, W_pad - pw + 1, sw)
    
    # If steps don't cover the end, add one last step to ensure coverage
    # Actually, range stops before stop.
    # We need to make sure we cover the last part.
    # A simple way is to check if the last patch covers the end.
    # The current range logic might miss the end if (D_pad - pd) % sd != 0.
    # We should add [D_pad - pd] to the list if not present.
    
    d_steps = list(d_steps)
    if d_steps[-1] != D_pad - pd: d_steps.append(D_pad - pd)
    
    h_steps = list(h_steps)
    if h_steps[-1] != H_pad - ph: h_steps.append(H_pad - ph)
    
    w_steps = list(w_steps)
    if w_steps[-1] != W_pad - pw: w_steps.append(W_pad - pw)
    
    model.eval()
    with torch.no_grad():
        for d in tqdm(d_steps, leave=False, desc="Slicing Z"):
            for h in h_steps:
                for w in w_steps:
                    # Extract patch
                    patch = image[:, d:d+pd, h:h+ph, w:w+pw].unsqueeze(0).to(DEVICE) # (1, C, D, H, W)
                    
                    # Predict
                    pred = model(patch)
                    pred = torch.sigmoid(pred)
                    
                    # Accumulate
                    output[:, d:d+pd, h:h+ph, w:w+pw] += pred.squeeze(0)
                    count_map[:, d:d+pd, h:h+ph, w:w+pw] += 1.0

    # Avoid division by zero
    count_map[count_map == 0] = 1.0
    output = output / count_map
    
    # Crop back to original size (important because we padded)
    output = output[:, :D, :H, :W]
    
    return output

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Load model
    print("Loading model...")
    # Model definition inside model.py is already updated to Residual UNet
    model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        print("Model loaded.")
    else:
        print("Warning: Model checkpoint not found. Using random initialization.")

    image_paths = sorted(glob.glob(os.path.join(TEST_IMG_DIR, '*.nii.gz')))
    
    for img_path in tqdm(image_paths, desc="Processing files"):
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        # Load image
        img_obj = nib.load(img_path)
        img_data = img_obj.get_fdata().astype(np.float32)
        affine = img_obj.affine
        
        # Normalize - same as training (Percentile)
        p99 = np.percentile(img_data, 99.5)
        if p99 > 0:
            img_data = np.clip(img_data, 0, p99) / p99
            
        # To Tensor: (1, D, H, W) -> Permute -> (1, H, W, D) -> Permute -> (C, D, H, W)
        # Wait, nibabel loads (H, W, D) = (1024, 1024, 92)
        # My dataset used: from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)
        # -> (1, 92, 1024, 1024)
        
        tensor_img = torch.from_numpy(img_data).unsqueeze(0) # (1, H, W, D)
        tensor_img = tensor_img.permute(0, 3, 1, 2) # (1, D, H, W)
        
        # Inference
        pred_tensor = sliding_window_inference(model, tensor_img, patch_size=PATCH_SIZE, stride=(32, 64, 64))
        
        # Threshold
        pred_mask = (pred_tensor > 0.5).float().cpu().numpy()
        
        # Inverse permute: (1, D, H, W) -> (1, H, W, D) -> squeeze -> (H, W, D)
        pred_mask = np.transpose(pred_mask, (2, 3, 1, 0)).squeeze() # (H, W, D)
        
        # Save
        save_path = os.path.join(OUTPUT_DIR, filename.replace(".nii.gz", "_pred.nii.gz"))
        nib.save(nib.Nifti1Image(pred_mask.astype(np.uint8), affine), save_path)
        print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
