import os

import torch
import numpy as np
import cv2

from utils import create_classify_model, create_segment_model, preprocess_image
from utils import extract_crops_with_stride

class Segment:	
	"""Frame segmentation"""

	def __init__(self):
		"""init model"""
		self.class_model = create_classify_model("weights/mobilenet_class.pth") 
		self.segment_model = create_segment_model("weights/r101best.pth")


	def predict(self, frame):
		"""segment frame and return % of bad road"""
		
		crops = self._predict_segment(frame)
		persent = self._predict_class(crops)
		
		return persent

	# return a lot of crops orig image
	def _predict_segment(self, frame):
		tensor_frame = preprocess_image(frame, 'segment')
		with torch.no_grad():
			output = self.segment_model(tensor_frame)

			pred = output.squeeze().argmax(0).cpu().numpy()

			bin_mask = (pred > 0).astype(np.uint8)*255

			h, w, _ = frame.shape

			resized_mask = cv2.resize(bin_mask, (w, h), cv2.INTER_NEAREST)

			segmented_image = np.zeros_like(frame)

			segmented_image[resized_mask==255] = frame[resized_mask==255]

			croped = extract_crops_with_stride(segmented_image)

		return croped


	def _predict_class(self, crops):
		count = 0
		len_crops = len(crops)
		with torch.no_grad():
			for frame in crops:
				tensor_frame = preprocess_image(frame, 'classify')
			
				output = self.class_model(tensor_frame)
				res = output.item()
				if res > 0.9:
					count += 1
				elif res > 0.5 and res < 0.9:
					count += 0.5
				
				if res < 0.1:
					count -= 3
				elif res < 0.5 and res > 0.1:
					count -= 1.5


		if len_crops == 0: 
			return 0

		return count

