import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

class MRADataset(Dataset):
    def __init__(self, root_dir, mode='train', patch_size=(128, 128, 64)):
        self.root_dir = root_dir
        self.mode = mode
        self.patch_size = patch_size
        
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, 'raw', '*.nii.gz')))
        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, 'gt', '*.nii.gz')))
        
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch in image and mask count"
        if len(self.image_paths) == 0:
            print(f"Warning: No files found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def normalize(self, image):
        # Robust normalization using percentiles
        # Clip top 0.5% outliers
        p99 = np.percentile(image, 99.5)
        # Avoid division by zero
        if p99 == 0:
            return image
        image = np.clip(image, 0, p99)
        return image / p99

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load NIfTI
        image_obj = nib.load(image_path)
        mask_obj = nib.load(mask_path)
        
        image = image_obj.get_fdata().astype(np.float32)
        mask = mask_obj.get_fdata().astype(np.float32)
        
        # Binary mask (ensure 0 and 1)
        mask = (mask > 0.5).astype(np.float32)
        
        # Normalize
        image = self.normalize(image)

        if self.mode == 'train':
            # Random Crop with Foreground Oversampling
            # We want to ensure we sample patches with vessels
            d, h, w = image.shape  
            pd, ph, pw = self.patch_size
            
            # Simple rejection sampling to find a patch with vessels
            # Try 10 times to find a positive patch, else fallback to random
            # This is less memory intensive than pre-calculating indices
            
            for _ in range(10):
                # Random coordinates
                x = np.random.randint(0, max(1, d - pd + 1))
                y = np.random.randint(0, max(1, h - ph + 1))
                z = np.random.randint(0, max(1, w - pw + 1))
                
                # Check if patch has vessels
                patch_mask = mask[x:x+pd, y:y+ph, z:z+pw]
                if patch_mask.sum() > 100: # At least 100 voxels of vessel
                    break
            
            # If loop finishes without break, we use the last random coords
            # (which is fine, we don't want 100% positive patches anyway)
            
            image = image[x:x+pd, y:y+ph, z:z+pw]
            mask = mask[x:x+pd, y:y+ph, z:z+pw]

            
        else:
            # Test/Val mode: Return full volume? 
            # Or center crop for simplicity in validation loop
            # Returning full volume might cause issues in batching if sizes differ slightly (though they seem consistent)
            # For validation, let's just take a center crop to track metrics quickly.
            # Full inference should be done in predict.py with sliding window.
            d, h, w = image.shape
            pd, ph, pw = self.patch_size
            
            # Center crop
            x = (d - pd) // 2
            y = (h - ph) // 2
            z = (w - pw) // 2
            
            # Ensure non-negative
            x, y, z = max(0, x), max(0, y), max(0, z)
            
            # Use min to handle if image smaller than patch (though handled by padding logic above if reused)
            # For simplicity assume consistent size or just crop
            image = image[x:x+pd, y:y+ph, z:z+pw]
            mask = mask[x:x+pd, y:y+ph, z:z+pw]

        image = torch.from_numpy(image).float().unsqueeze(0) # (1, H, W, D)
        mask = torch.from_numpy(mask).float().unsqueeze(0) # (1, H, W, D)
        
        # Permute to (C, D, H, W) for PyTorch Conv3d
        # Input is (1, 128, 128, 64) -> (1, 64, 128, 128)
        image = image.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)
        
        return image, mask
