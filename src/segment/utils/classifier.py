import os
import threading
import argparse
import queue

import torch
import cv2
from lgbt import lgbt
import numpy as np
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import DeepLabV3Plus, DeepLabV3
import cv2

from image_utils import extract_crops_with_stride, calculate_black_pixels_percentage, preprocess_image

def create_deeplabv3plus(num_classes):
	model = DeepLabV3Plus(encoder_name="resnet101", classes=num_classes)
    
	return model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = '../weights/r101best.pth'       
NUM_CLASSES = 25                               
RESIZE_SIZE = (1024, 1024)
MIN_CLUST = 500             


test_transform = A.Compose([
	A.Resize(*RESIZE_SIZE),
	A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	ToTensorV2()
])

def load_model():
	model = create_deeplabv3plus(num_classes=NUM_CLASSES).to(DEVICE)
	model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
	model.eval()
	return model



def parse_arg():
	parser = argparse.ArgumentParser(description='Find images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('-i', '--input', type=str, required=True, help='Путь к входной папке с изображениями')
	parser.add_argument('-o', '--output', type=str, default='output', help='Путь для сохранения результатов')
	parser.add_argument('-c', '--crop', type=str, default='false', help="Режим кропинга папки")

	args = parser.parse_args()
	if not os.path.isdir(args.input):
		raise NotADirectoryError("Not correct path")
	
	args.output = os.path.abspath(args.output)
	os.makedirs(args.output, exist_ok=True)
    
	return args

queue = queue.Queue(maxsize=1000)
condition = threading.Condition()
model = load_model()

def segment_mask(image_rgb):
	original_image = image_rgb
	image_tensor = preprocess_image(image_rgb)
	
	with torch.no_grad():
		# make predict
		output = model(image_tensor)

		# putting together all classes
		pred = output.squeeze().argmax(0).cpu().numpy()
		combined_image = (pred > 0).astype(np.uint8) * 255

		# converting to image
		mask_pil = Image.fromarray(combined_image)
		w, h, _ = original_image.shape

		# resize mask (by default 1024x1024)
		mask_resized = mask_pil.resize((h,w), Image.NEAREST)
		mask_resized = np.array(mask_resized)

		# getting mask from original image
		segment = np.zeros_like(original_image)
		segment[mask_resized==255] = original_image[mask_resized==255]

	return segment

def logic_crop(image_rgb, path_to_save, image_name):
	if image_rgb is None:
		return
	masked = segment_mask(image_rgb)
	croped_images = extract_crops_with_stride(masked)
	ind = 0
	for image in croped_images:
		name = f'{image_name}_{ind}.jpg'
		new_path = os.path.join(path_to_save, name)
		cv2.imwrite(new_path, image)
		ind += 1


def logic_common(image_rgb, path_to_save, image_name):
	file_name = os.path.join(path_to_save, image_name)
	cv2.imwrite(file_name, image_rgb)


def saver_crop(path):
	output_path = path
	crop_path = os.path.join(output_path, "crops")
	os.makedirs(crop_path, exist_ok=True)

	while True:
		with condition:
			if queue.empty():
				condition.wait()
			res, image_name, image_rgb = queue.get()
			if res:
				break

		logic_crop(image_rgb, crop_path, image_name)

def saver_common(path):
	output_path = path
	good_path = os.path.join(output_path, "good")
	bad_path = os.path.join(output_path, "bad")
	os.makedirs(good_path, exist_ok=True)
	os.makedirs(bad_path, exist_ok=True)

	while True:
		with condition:
			if queue.empty():
				condition.wait()
			res, image_name, image_rgb = queue.get()

			if res == 'exit':
				while not queue.empty():
					res, image_name, image_rgb = queue.get()
					logic_common(image_rgb, good_path, image_name)
				break
			elif res == 'good':
				path_to_save = good_path
			elif res == 'bad':
				path_to_save = bad_path

		logic_common(image_rgb, path_to_save, image_name)


def main():
	global saver

	args = parse_arg()
	img_path = args.input
	output_path = args.output
	crop = args.crop

	if crop == 'true':
		saver = saver_crop 
	else:
		saver = saver_common

		win_name = "q=GOOD e=BAD"
		cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(win_name, 224, 224) 

		position = (50, 50)
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 1
		color = (255, 0, 0)  
		thickness = 1


	thread = threading.Thread(target=saver, args=(output_path,))
	thread.start()

	img_list = os.listdir(img_path)

	if len(img_list) == 0:
		raise FileNotFoundError("No images in the input folder")


	if crop == 'true':
		for img_name in lgbt(img_list, desc='progress', mode='mex', hero='sakura'):
			img_full_path = os.path.join(img_path, img_name)
			img_bgr = cv2.imread(img_full_path)
			img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

			with condition:
				queue.put((False, img_name, img_rgb))
				condition.notify()

		with condition:
			queue.put((True, None, None))
			condition.notify()
		thread.join()
	else:
		res = None 
		count = 0
		for img_name in img_list:
			img_full_path = os.path.join(img_path, img_name)
			img_bgr = cv2.imread(img_full_path)
			img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
			img_bgr = cv2.resize(img_bgr, (224,244))
			cv2.putText(img_bgr, str(count), position, font, font_scale, color, thickness)
			cv2.imshow(win_name, img_bgr)

			key = cv2.waitKeyEx(0)
			if key == 27: # end (ECS)
				res = 'exit'
			elif key == ord('q'): # good marking
				res = 'good'
			elif key == ord('e'): # bad marking
				res = 'bad'

			with condition:
				queue.put((res, img_name, img_rgb))
				condition.notify()
			count += 1

			if res == 'exit':
				break

		with condition:
			queue.put(("exit", None, None))
			condition.notify()
		
		thread.join()
		cv2.destroyAllWindows()

# --- Основной процесс ---
if __name__ == "__main__":
	main()