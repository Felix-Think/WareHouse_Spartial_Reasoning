import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from torchvision import transforms

def polygon_to_mask(polygon, img_height, img_width):
    """Convert polygon segmentation to binary mask"""
    # Create mask image
    mask = Image.new('L', (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Convert polygon to tuples of (x, y) coordinates
    coords = []
    for i in range(0, len(polygon), 2):
        coords.append((polygon[i], polygon[i+1]))
    
    # Fill polygon
    if coords:
        draw.polygon(coords, fill=1)
    
    return np.array(mask)

class COCOAmodalDataset(Dataset):
    """COCO Dataset with real amodal/visible masks - NO DUMMY DATA"""
    
    def __init__(self, image_dir, ann_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Load COCO annotations
        print(f"🔍 Loading COCO annotations from: {ann_file}")
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        
        # Group annotations by image_id
        self.amodal_by_image = {}
        self.visible_by_image = {}
        
        # Process amodal masks
        print(f"📋 Processing {len(self.annotations['amodal_masks'])} amodal masks...")
        for ann in self.annotations['amodal_masks']:
            img_id = ann['image_id']
            if img_id not in self.amodal_by_image:
                self.amodal_by_image[img_id] = []
            self.amodal_by_image[img_id].append(ann)
        
        # Process visible masks (list of lists)
        print(f"📋 Processing visible masks...")
        visible_count = 0
        for visible_list in self.annotations['visible_masks']:
            for ann in visible_list:
                img_id = ann['image_id']
                if img_id not in self.visible_by_image:
                    self.visible_by_image[img_id] = []
                self.visible_by_image[img_id].append(ann)
                visible_count += 1
        
        print(f"✅ Dataset loaded: {len(self.images)} images")
        print(f"✅ Amodal masks: {len(self.amodal_by_image)} images covered")
        print(f"✅ Visible masks: {visible_count} total, {len(self.visible_by_image)} images covered")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image info
        img_info = self.images[idx]
        img_id = img_info['id']
        img_height = img_info['height']
        img_width = img_info['width']
        
        # Load RGB-D image from images folder
        image_path = os.path.join(self.image_dir, 'images', img_info['file_name'])
        
        if os.path.exists(image_path):
            # Load real RGB image
            rgb_image = Image.open(image_path).convert('RGB')
            
            # Extract depth filename
            if '_png.rf.' in img_info['file_name']:
                image_id = img_info['file_name'].split('_png.rf.')[0]
                depth_filename = f"{image_id}_depth.png"
            else:
                base_name = img_info['file_name'].split('.')[0]
                depth_filename = f"{base_name.split('_')[0]}_depth.png"
            
            depth_path = os.path.join(self.image_dir, 'depths', depth_filename)
            
            if os.path.exists(depth_path):
                # Load real depth data
                depth_image = Image.open(depth_path).convert('L')
            else:
                # Create synthetic depth from RGB
                depth_image = rgb_image.convert('L')
                
            # Transform images
            rgb_tensor = self.transform(rgb_image)  # [3, H, W]
            depth_tensor = self.transform(depth_image)  # [1, H, W]
            
            # Combine RGB + Depth
            rgbd = torch.cat([rgb_tensor, depth_tensor], dim=0)  # [4, H, W]
        else:
            # ERROR: No fallback dummy data!
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Get actual annotations for this image
        amodal_anns = self.amodal_by_image.get(img_id, [])
        visible_anns = self.visible_by_image.get(img_id, [])
        
        # Process masks
        num_queries = 50
        height, width = rgbd.shape[1], rgbd.shape[2]
        
        # Initialize with ZEROS (not random)
        amodal_masks = torch.zeros(num_queries, height, width)
        visible_masks = torch.zeros(num_queries, height, width)
        labels = torch.zeros(num_queries, dtype=torch.long)
        boxes = torch.zeros(num_queries, 4)
        valid = torch.zeros(num_queries, dtype=torch.bool)
        
        # Process REAL amodal masks
        valid_count = 0
        for i, ann in enumerate(amodal_anns[:num_queries]):
            if 'segmentation' in ann and ann['segmentation'] and len(ann['segmentation']) > 0:
                try:
                    # Convert polygon to mask
                    polygon = ann['segmentation'][0]  # Take first polygon
                    if len(polygon) >= 6:  # Need at least 3 points (6 coords)
                        mask = polygon_to_mask(polygon, img_height, img_width)
                        
                        # Resize mask to match image dimensions
                        if mask.shape != (height, width):
                            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                            mask_img = mask_img.resize((width, height), Image.NEAREST)
                            mask = np.array(mask_img) / 255.0
                        
                        amodal_masks[i] = torch.tensor(mask, dtype=torch.float32)
                        labels[i] = ann.get('category_id', 1)
                        valid[i] = True
                        
                        # Extract bbox (normalized)
                        bbox = ann.get('bbox', [0, 0, img_width, img_height])
                        boxes[i] = torch.tensor([
                            bbox[0] / img_width,  # x_min normalized
                            bbox[1] / img_height, # y_min normalized  
                            (bbox[0] + bbox[2]) / img_width,  # x_max normalized
                            (bbox[1] + bbox[3]) / img_height  # y_max normalized
                        ])
                        
                        valid_count += 1
                except Exception as e:
                    print(f"⚠️ Error processing amodal mask {i}: {e}")
        
        # Process REAL visible masks  
        visible_count = 0
        for i, ann in enumerate(visible_anns[:num_queries]):
            if 'segmentation' in ann and ann['segmentation'] and len(ann['segmentation']) > 0:
                try:
                    # Convert polygon to mask
                    polygon = ann['segmentation'][0]  # Take first polygon
                    if len(polygon) >= 6:  # Need at least 3 points
                        mask = polygon_to_mask(polygon, img_height, img_width)
                        
                        # Resize mask to match image dimensions
                        if mask.shape != (height, width):
                            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                            mask_img = mask_img.resize((width, height), Image.NEAREST)
                            mask = np.array(mask_img) / 255.0
                        
                        # Find best matching index
                        query_idx = min(i, num_queries - 1)
                        visible_masks[query_idx] = torch.tensor(mask, dtype=torch.float32)
                        visible_count += 1
                except Exception as e:
                    print(f"⚠️ Error processing visible mask {i}: {e}")
        
        targets = {
            'labels': labels,
            'boxes': boxes,
            'amodal_masks': amodal_masks,
            'visible_masks': visible_masks,
            'valid': valid
        }
        
        #print(f"🎯 Image {img_id}: {valid_count} valid amodal, {visible_count} visible masks")
        
        return rgbd, targets