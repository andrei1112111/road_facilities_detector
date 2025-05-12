import os
import shutil

import numpy as np

def split_list(lst, n):
	return np.array_split(lst, n)

path = r"C:\Users\user\Downloads\train\images"
path_out = r"C:\NSU\2k\2s\Project\dataset"

images = os.listdir(path)
folder_num = 0

parts_of_images = split_list(images, 18)

for part in parts_of_images:

	list_part = part.tolist()
	part_path = os.path.join(path_out, f'part{folder_num}')
	os.makedirs(part_path, exist_ok=True)

	for image_name in list_part:
		
		source_path = os.path.join(path, image_name)
		target_path = os.path.join(path_out, f'part{folder_num}')
		shutil.move(source_path, target_path)
	folder_num += 1
