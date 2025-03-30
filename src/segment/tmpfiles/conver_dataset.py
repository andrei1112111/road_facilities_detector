import os
import shutil


file_structure = {
	'ann_train': 'ade/ADEChallengeData2016/annotations/training',
	'ann_val': 'ade/ADEChallengeData2016/annotations/validation',
	'img_train': 'ade/ADEChallengeData2016/images/training',
	'img_val': 'ade/ADEChallengeData2016/images/validation',
	} 

def create_dirs():
	for dir in file_structure:
		os.makedirs(file_structure[dir])


def move_files(source_dir, target_dir):

	train_path = os.path.join(source_dir, 'train')
	valid_path = os.path.join(source_dir, 'valid')

	all_path = { 'training': train_path,'validation':valid_path }

	for path in all_path:
		for file in os.listdir(all_path[path]):
			path_from = os.path.join(all_path[path], file)
			path_to = ""
			print("from")
			print(path_from)

			if file.lower().endswith(".jpg"):
				path_to = os.path.join(target_dir,'ADEChallengeData2016', 'images', path, file)
				print("to")	
				print(path_to)
				shutil.copy(path_from, path_to)
					
			if file.lower().endswith(".png"):
				new_name = file.replace('_mask', "")
				path_to = os.path.join(target_dir,'ADEChallengeData2016', 'annotations', path, new_name)
				print("to")	
				print(path_to)
				shutil.copy(path_from, path_to)


def main():
	try:
		create_dirs()
	except FileExistsError:
		pass

	move_files('dataset2', 'ade')


if __name__ == "__main__":
	main()