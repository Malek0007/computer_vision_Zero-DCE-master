import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_DIR = os.path.join(DATA_DIR, 'result')
SNAPSHOT_PATH = os.path.join(BASE_DIR, 'snapshots', 'Epoch99.pth')


def lowlight(image_path):
	device = torch.device(
    'mps' if torch.backends.mps.is_available() else 'cpu'
)
	if torch.cuda.is_available():
		os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)

	data_lowlight = (np.asarray(data_lowlight)/255.0)
	data_lowlight = torch.from_numpy(data_lowlight).float()$
	data_lowlight = data_lowlight.permute(2,0,1).unsqueeze(0).to(device)

	DCE_net = model.enhance_net_nopool().to(device)
	DCE_net.load_state_dict(torch.load(SNAPSHOT_PATH, map_location=device))
	start = time.time()
	_,enhanced_image,_ = DCE_net(data_lowlight)

	end_time = (time.time() - start)
	print(end_time)
	result_path = image_path.replace(os.path.join(BASE_DIR, 'data', 'test_data'), os.path.join(BASE_DIR, 'data', 'result'))
	if not os.path.exists(os.path.dirname(result_path)):
		os.makedirs(os.path.dirname(result_path), exist_ok=True)

	torchvision.utils.save_image(enhanced_image.detach().cpu(), result_path)

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = os.path.join(DATA_DIR, 'test_data')
		file_list = os.listdir(filePath)

		for file_name in file_list:
			test_list = glob.glob(os.path.join(filePath, file_name, '*'))
			for image in test_list:
				print(image)
				lowlight(image)

		

