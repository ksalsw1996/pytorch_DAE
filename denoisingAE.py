import os
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as d_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
is_cuda = torch.cuda.is_available()
parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=40, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--model_dir', type=str, default='model_dir', help='model_saving directory')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--noise_t', type=str, default='MASK', help='Type of noise')
parser.add_argument('--noise_r', type=int, default=10, help='Ratio of noise')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--output_dir', type=str, default='result_SDAE', help='Directory for result')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

opt = parser.parse_args()
try:
    os.makedirs(opt.output_dir)
except OSError:
    pass

transform=transforms.Compose([
	transforms.Scale(opt.imageSize),
	transforms.ToTensor(),
	transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

def salt_and_pepper(X, prop):
        X_clone=X.clone().view(-1, 1)
        num_feature=X_clone.size(0)
        mn=X_clone.min()
        mx=X_clone.max()
        indices=np.random.randint(0, num_feature, int(num_feature*prop))
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
        return X_clone.view(X.size())

class TestDataset(d_utils.Dataset):
	def __init__(self, root='./dataset/TEST_REAL', transform=None, noise_t = 'MASK', noise_r=10):
		self.noise_t = noise_t
		self.root=root
		self.f_list=os.listdir(self.root)
		self.pair_data='./dataset/TEST_'+noise_t+'_'+str(noise_r)
		self.noise_list=os.listdir(self.pair_data)
		self.transform=transform
	def __len__(self):
		return len(self.f_list)
	def __getitem__(self, idx):
		Img = os.path.join(self.root, self.f_list[idx])
		Img = Image.open(Img).convert('L')
		Noise_Img = os.path.join(self.pair_data, 'noise_data_'+self.f_list[idx].split('_')[-1])
		Noise_Img = Image.open(Noise_Img).convert('L')
		if self.transform is not None :
			Img=self.transform(Img)
			Noise_Img=self.transform(Noise_Img)
		return {'image' : Img, 'noise' : Noise_Img}

Train_dataset = dset.MNIST(root='./dataset/TRAIN_DATA', train=True, download=True, transform=transform)
Test_dataset = TestDataset(root='./dataset/TEST_REAL', noise_t='MASK', noise_r='20', transform=transform)
train_loader = d_utils.DataLoader(Train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = d_utils.DataLoader(Test_dataset, batch_size=64, shuffle=True, num_workers=2)

class _SDAE(nn.Module):
	def __init__(self, ngpu):
		super(_SDAE, self).__init__()
		self.ngpu=ngpu
		self.encoder = nn.Sequential(
			nn.Linear(784, 400),
			nn.SELU(),
			nn.Linear(400, 200),
			nn.SELU(),
			nn.Linear(200, 100),
			nn.SELU())
		self.decoder = nn.Sequential(
			nn.Linear(100, 200),
			nn.SELU(),
			nn.Linear(200, 400),
			nn.SELU(),
			nn.Linear(400, 784))
		self.main = nn.Sequential(
			self.encoder,
			self.decoder)
	def forward(self, input):
		size=input.size()
		if is_cuda and self.ngpu>1 :
			output=nn.parallel.data_parallel(self.main, input.view(-1, 784), range(self.ngpu))
		else :
			output=self.main(input.view(-1, 784))
		return output.view(size)

SDAE=_SDAE(opt.ngpu)
criterion = nn.MSELoss()
input_ori = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)
input_noise = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize)

if is_cuda:
	SDAE.cuda()
	criterion.cuda()
	input_ori.cuda()
	input_noise.cuda()

if opt.noise_t == 'MASK':
	noise_fn=masking
else :
	noise_fn = salt_and_pepper

optimizer = optim.Adam(SDAE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.epoch):
	print "Epoch : %d" %(epoch+1)
	p_error=[]
	p_real=None
	p_noise=None
	p_restored=None
	for i, data in enumerate(train_loader, 0):
		SDAE.zero_grad()
		data, _ = data
		input_ori.resize_as_(data).copy_(data)
		input_v=Variable(input_ori)
		noise_data = noise_fn(data, float(opt.noise_r)/100)
		input_noise.resize_as_(noise_data).copy_(noise_data)
		input_nv=Variable(input_noise)
		if is_cuda:
			input_v=input_v.cuda()
			input_nv=input_nv.cuda()
		output=SDAE(input_nv)
		error=criterion(output, input_v)
		error.backward()
		optimizer.step()
	for i, data in enumerate(test_loader, 0):
		SDAE.zero_grad()
		real = data['image']
		noise = data['noise']
		input_ori.resize_as_(real).copy_(real)
		input_v=Variable(input_ori)
		input_noise.resize_as_(noise).copy_(noise)
		input_nv=Variable(noise)
		if is_cuda:
			input_v=input_v.cuda()
			input_nv=input_nv.cuda()
		output=SDAE(input_nv)
		error=criterion(output,input_v)
		p_error+=error.data.cpu().numpy().tolist()
		if real.size()[0]==opt.batchSize :
			p_real=real
			p_noise=noise
			p_restored=output
		if i%100==0:
			print "[%3d:%3d] loss : %.4f" %(i, len(test_loader), p_error[0])
	print "Error : %f" %np.mean(p_error)
	vutils.save_image(p_real,
		'%s/real_samples_epoch_%03d.png' % (opt.output_dir, epoch),
		normalize=True)
	vutils.save_image(p_noise,
		'%s/noise_samples_epoch_%03d.png' % (opt.output_dir, epoch),
		normalize=True)
	vutils.save_image(p_restored.data,
		'%s/restored_samples_epoch_%03d.png' % (opt.output_dir, epoch),
		normalize=True)
