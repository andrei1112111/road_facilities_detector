import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from test import show_mask
from test import get_predicted


def main():
	global device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	img = get_predicted("test_img\image.jpg")
	print(img)

if __name__ == "__main__":
	main()
