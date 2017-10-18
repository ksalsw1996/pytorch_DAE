import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.utils.data as d_utils
import torchvision.utils as vutil
import numpy as np
torch.manual_seed(int(time.time()))
transform = transforms.Compose([
	transforms.Scale(28),
	transforms.ToTensor(),
	transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
dataset = dset.MNIST(root='./MNIST/', download=True, train=False, transform=transform)
dataloader = d_utils.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

def salt_and_pepper(X, prop):
	X_clone=X.clone().view(-1, 1)
	num_feature=X_clone.size(0)
	mn=X_clone.min()
	mx=X_clone.max()
	indices=np.random.randint(0, num_feature, int(num_feature*prop))
#	print indices
	for elem in indices :
		if np.random.random() < 0.5 :
			X_clone[elem]=mn
		else :
			X_clone[elem]=mx
	return X_clone.view(X.size())

def masking(X, p):
	X_clone=X.clone()
	lenx=X_clone.size(2)
	leny=X_clone.size(3)
	for i in range(X_clone.size(0)):
		maskx=np.random.uniform(p, 1, 1)
		masky=p/maskx
		maskx=int(maskx*lenx)
		masky=int(masky*leny)
		idx=np.random.randint(0,lenx-maskx, 1)
		idy=np.random.randint(0,leny-masky, 1)
		for j in range(idx, idx+maskx):
			for k in range(idy, idy+masky):
				X_clone[i][0][j][k]=0
	return X_clone
	

for i, data in enumerate(dataloader, 0):
#	print i
	dat, _ = data
#	noise_data=salt_and_pepper(dat, 0.30)
	noise_data=masking(dat,0.3)
	#print noise_data.size()
#	vutil.save_image(dat, './dataset/TEST_REAL/real_data_%d.png' %i, normalize=True)
	vutil.save_image(noise_data, './dataset/TEST_MASK_30/noise_data_%d.png' %i, normalize=True)

