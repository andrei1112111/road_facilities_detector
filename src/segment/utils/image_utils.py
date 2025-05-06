import numpy as np
import albumentations as A

from .models import device

def calculate_black_pixels_percentage(img):
    black_mask = np.all(img == [0, 0, 0], axis=-1)
    black_pixels = np.sum(black_mask)
    total_pixels = img.shape[0] * img.shape[1]
    percentage_black = (black_pixels / total_pixels) * 100
    
    return percentage_black

def extract_crops_with_stride(img, crop_size=224, stride=224):
	h, w, _ = img.shape
	crops = []

	for y in range(0, h - crop_size + 1, stride):
		for x in range(0, w - crop_size + 1, stride):
			crop = img[y:y+crop_size, x:x+crop_size]
			if (calculate_black_pixels_percentage(crop) < 85):
				crops.append(crop)

	return crops

segment_transform = A.Compose([
	A.Resize(height=1024,width=1024),
	A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	A.ToTensorV2()
])

classify_transform  = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2()  
])

transforms = { 'segment': segment_transform, 'classify': classify_transform }

def preprocess_image(image_rgb, type='segment'):
	transform = transforms[type]

	transformed = transform(image=image_rgb)
	image_tensor = transformed["image"].unsqueeze(0).to(device)

	return image_tensor