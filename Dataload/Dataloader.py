import cv2
import numpy as np
import PIL.Image as Image
from typing import List, Dict, Any
from pycocotools import mask as maskUtils
import os
import json

def segm_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    """
    Chuyển đổi segmentation polygon thành binary mask
    """
    if isinstance(segmentation, list):
        # Polygon format
        rles = maskUtils.frPyObjects(segmentation, height, width)
        rle = maskUtils.merge(rles)
    else:
        # RLE format
        rle = segmentation
    
    return maskUtils.decode(rle)

def create_class_object_matrix(image_id: str, coco_json: str) -> np.ndarray:
    """
    Tạo ma trận 512x512xN với class IDs
    
    Args:
        image_id: Tên ảnh (không có extension)
        coco_json: Đường dẫn tới file COCO annotations
        output_dir: Thư mục output
    
    Returns:
        Ma trận numpy shape (512, 512, N) với values = class_id
    """
    
    # Đọc COCO data
    with open(coco_json, 'r') as f:
        data = json.load(f)
    
    # Tìm image info
    target_image_info = None
    for img in data['images']:
        img_basename = os.path.splitext(img['file_name'])[0]
        if img_basename == image_id:
            target_image_info = img
            break
    
    if not target_image_info:
        print(f"❌ Không tìm thấy ảnh: {image_id}")
        return None
    
    img_width = target_image_info['width']
    img_height = target_image_info['height']
    
    # Lấy tất cả annotations của ảnh này
    image_annotations = [ann for ann in data['annotations'] 
                        if ann['image_id'] == target_image_info['id']]
    
    if not image_annotations:
        print(f"❌ Không có annotations cho ảnh: {image_id}")
        return None
    
    num_objects = len(image_annotations)
    print(f"✅ Ảnh {image_id}: {img_width}x{img_height}, {num_objects} objects")
    
    # Tạo ma trận 3D: (height, width, num_objects)
    object_matrix = np.zeros((img_height, img_width, num_objects), dtype=np.uint8)
    
    # Xử lý từng annotation
    for idx, ann in enumerate(image_annotations):
        # Tạo mask từ segmentation
        segmentation = ann['segmentation']
        mask = segm_to_mask(segmentation, img_height, img_width)
        
        # Lấy class ID của object
        class_id = ann['category_id']
        
        # Gán mask với giá trị = class_id vào channel tương ứng
        object_matrix[:, :, idx] = np.where(mask, class_id, 0).astype(np.uint8)    
    return object_matrix

def getDepth(instance_mask, filedepth):
    coords = (instance_mask != 0).astype(np.uint8)
    depth = np.array(Image.open(filedepth))
    depth = np.resize(depth, (512, 512))
    vals = depth[coords == 1]
    mean_depth = np.mean(vals) if vals.size > 0 else 0
    return mean_depth
def arrange_occlusion(matrices, filedepth):
    H, W, N = matrices.shape
    depths = []
    class_ids = []

    for i in range(N):
        instance_mask = matrices[:, :, i]

        # lấy class id từ giá trị khác 0 trong mask
        uniq_vals = np.unique(instance_mask)
        uniq_vals = uniq_vals[uniq_vals != 0]  # bỏ background
        if len(uniq_vals) > 0:
            cid = int(uniq_vals[0])   # giả sử 1 mask chỉ có 1 class
        else:
            cid = -1  # background hoặc rỗng

        # tính depth
        if cid == 1:  # buffer luôn dưới cùng
            d = float("inf")
        else:
            d = getDepth(instance_mask, filedepth)

        depths.append((i, d))
        class_ids.append(cid)

    # sắp xếp theo depth tăng dần
    depths_sorted = sorted(depths, key=lambda x: x[1])

    # tạo mảng mới đã sắp xếp
    sorted_matrices = np.zeros_like(matrices)
    sorted_class_ids = []

    for new_idx, (old_idx, _) in enumerate(depths_sorted):
        sorted_matrices[:, :, new_idx] = matrices[:, :, old_idx]
        sorted_class_ids.append(class_ids[old_idx])

    return sorted_matrices

def split_masks(sorted_matrices):
    H, W, N = sorted_matrices.shape
    mask_images = []

    for i in range(N):
        mask = sorted_matrices[:, :, i]
        # chuẩn hóa: 0 background, 255 foreground
        mask_img = (mask > 0).astype(np.uint8) * 255
        mask_images.append(mask_img)

    return mask_images

def save_masks(mask_images: List[np.ndarray], output_dir: str):
    for idx, m in enumerate(mask_images):
        Image.fromarray(m).save(f"{output_dir}/mask_{idx}.png")


def getVisibleObjects(matrix):
    H, W, N = matrix.shape
    # khởi tạo array rỗng có shape (H, W, 0)
    visible_objects = np.zeros((H, W, 0), dtype=np.uint8)

    for i in range(N):
        instance_mask = matrix[:, :, i]
        class_ids = np.unique(instance_mask)
        class_ids = class_ids[class_ids != 0]  # bỏ background

        if len(class_ids) == 0:
            continue
        cid = int(class_ids[0])

        # chuẩn hóa mask hiện tại
        instance_mask = (instance_mask > 0).astype(np.uint8) * cid

        # loại bỏ phần bị che bởi các mask trước đó
        for j in range(i):
            prev_mask = (matrix[:, :, j] > 0).astype(np.uint8)
            instance_mask = instance_mask * (1 - prev_mask)

        # thêm mask này vào channel cuối
        instance_mask = instance_mask[:, :, None]  # (H, W) -> (H, W, 1)
        visible_objects = np.concatenate((visible_objects, instance_mask), axis=2)

    return visible_objects



def get_json_data(coco_json: str) -> Dict[str, Any]:
    """
    Đọc file JSON COCO và trả về dữ liệu dưới dạng dictionary
    """
    with open(coco_json, 'r') as f:
        data = json.load(f)
    return data



def mask_to_polygon(mask):
    # mask: (H, W) uint8, 0/1 hoặc 0/255
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3: # at least 3 points
            contour = contour.reshape(-1, 2)
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    # Nếu có nhiều contour, chọn contour lớn nhất (thường là foreground object)
    if len(polygons) == 0:
        return []
    polygons = max(polygons, key=len)
    return polygons

def create_visible_polygon(img_id, file_name, visible_matrix, coco_json):
    data = get_json_data(coco_json)
    if visible_matrix is None:
        return []
    H, W, N = visible_matrix.shape
    # Sắp xếp theo độ sâu
    depth_path = file_name[:6]
    depth_file = f"/run/media/felix/Kingston/WareHouse_Spartial_Reasoning/Dataload/depths/{depth_path}_depth.png"
    sorted_matrix = arrange_occlusion(visible_matrix, depth_file)
        
    # Lấy các object nhìn thấy được
    visible_objects = getVisibleObjects(sorted_matrix)

    #tao annotations moi
    annotations = []
    for i in range(N):
        instance_mask = visible_objects[:, :, i]
        class_ids = np.unique(instance_mask)
        class_ids = class_ids[class_ids != 0]  # bo qua background

        if len(class_ids) == 0:
            continue
        cid = int(class_ids[0])
        # chuyển mask sang polygon
        polygons = mask_to_polygon(instance_mask)
        if len(polygons) == 0:
            continue
        annotation = {
            "id": len(annotations) + 1,  # ID mới
            "image_id": img_id,  # ID mới
            "category_id": cid,
            "segmentation": [polygons],
            "area": int(np.sum(instance_mask > 0)),
            "bbox": [int(np.min(np.where(instance_mask > 0)[1])), 
                     int(np.min(np.where(instance_mask > 0)[0])), 
                     int(np.max(np.where(instance_mask > 0)[1]) - np.min(np.where(instance_mask > 0)[1])), 
                     int(np.max(np.where(instance_mask > 0)[0]) - np.min(np.where(instance_mask > 0)[0]))],
            "iscrowd": 0
        }
        annotations.append(annotation)
    return annotations

def create_background_objs_polygon(img_id, visible_matrix, amodal_matrix, coco_json):
    """
    Tạo annotation polygon cho background objects mask từ visible và amodal mask.
    Args:
        image_id: tên ảnh (không extension)
        visible_matrix: numpy (H, W, N) mask visible
        amodal_matrix: numpy (H, W, N) mask amodal
        coco_json: đường dẫn file annotation COCO
    Returns:
        List annotation background_objs (COCO polygon format)
    """
    data = get_json_data(coco_json)
    if visible_matrix is None or amodal_matrix is None:
        return []
    H, W, N = visible_matrix.shape
    annotations = []
    for i in range(N):
        vis_mask = (visible_matrix[:, :, i] > 0).astype(np.uint8)
        amo_mask = (amodal_matrix[:, :, i] > 0).astype(np.uint8)
        # Tạo mask background: phần bị che khuất
        bg_mask = np.logical_and(amo_mask, np.logical_not(vis_mask)).astype(np.uint8)
        if np.sum(bg_mask) == 0:
            continue
        # Lấy class id
        class_ids = np.unique(amo_mask * amodal_matrix[:, :, i])
        class_ids = class_ids[class_ids != 0]
        if len(class_ids) == 0:
            continue
        cid = int(class_ids[0])
        # Chuyển mask sang polygon
        polygons = mask_to_polygon(bg_mask)
        if len(polygons) == 0:
            continue
        annotation = {
            "id": len(annotations) + 1,  # ID mới
            "image_id": img_id,
            "category_id": cid,
            "segmentation": [polygons],
            "area": int(np.sum(bg_mask > 0)),
            "bbox": [int(np.min(np.where(bg_mask > 0)[1])),
                     int(np.min(np.where(bg_mask > 0)[0])),
                     int(np.max(np.where(bg_mask > 0)[1]) - np.min(np.where(bg_mask > 0)[1])),
                     int(np.max(np.where(bg_mask > 0)[0]) - np.min(np.where(bg_mask > 0)[0]))],
            "iscrowd": 0
        }
        annotations.append(annotation)
    return annotations
    
def create_new_json_info(coco_json : str, output_json: str): 
    data = get_json_data(coco_json)
    # Chuẩn bị cấu trúc mới
    categories = data['categories']
    images = data['images']
    visible_masks = []
    back_ground_objs_mask = []
    # xu ly file anh , them depth vao images
    for i, img in enumerate(images):
        img_id = img['id']
        img_filename = img['file_name']
        height = img['height']
        width = img['width']
        depth_path = img_filename[:6]
        depth_file = f"{depth_path}_depth.png"
        img['depth_file'] = depth_file
        extra_info = img.get('extra', {})

        new_image = {
            "id": img_id,
            "file_name": img_filename,
            "height": height,
            "width": width,
            "depth_file": depth_file,
            "extra": extra_info
        }
        images[i] = new_image

        # Tao mot dict moi cho visible objects 
        file_name = os.path.splitext(img_filename)[0]
        object_matrix = create_class_object_matrix(file_name, coco_json)
        if object_matrix is None:
            print(f"❌ Không tạo được ma trận cho ảnh: {img_filename}")
            continue

        visible_masks_polygon = create_visible_polygon(img_id, file_name, object_matrix, coco_json)
        visible_masks.append(visible_masks_polygon)
        visible_object = getVisibleObjects(object_matrix)
        
        # Tạo polygon cho background objects
        amodal_matrix = object_matrix
        background_objs_polygon = create_background_objs_polygon(img_id, visible_object, amodal_matrix, coco_json)
        back_ground_objs_mask.append(background_objs_polygon)
    
    # xu ly annotations
    amodal_masks_polygon = data['annotations']
    
    new_data = {
        "categories": categories,
        "images": images,
        "gt_visible_masks": visible_masks,
        "gt_amodal_masks": amodal_masks_polygon,
        "gt_background_objs": back_ground_objs_mask
    }
    # Ghi dữ liệu mới vào file JSON
    with open(output_json, 'w') as f:
        json.dump(new_data, f, indent=4)
    print(f"✅ Đã tạo file JSON mới: {output_json}")
    

if __name__ == "__main__":
    # Ví dụ sử dụng
    #image_id = "000189_png.rf.22b550615cd58d13fb21c79b9777a11c"
    coco_json = "/run/media/felix/Kingston/WareHouse_Spartial_Reasoning/Dataload/train/_annotations.coco.json"
    
    #matrix = create_class_object_matrix(image_id, coco_json) # (H, W, N)

    #sorted_matrix = arrange_occlusion(matrix, "/run/media/felix/Kingston/WareHouse_Spartial_Reasoning/Dataload/depths/000189_depth.png")

    #visible_objects = getVisibleObjects(sorted_matrix)
    #os.makedirs("./outputs", exist_ok=True)
    #save_masks(split_masks(visible_objects), "./outputs")
    create_new_json_info(coco_json, "new_annotations.coco.json")