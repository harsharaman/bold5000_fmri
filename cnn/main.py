import argparse
import tqdm
import os
import yaml
import time
import numpy as np
import pandas as pd
import sys
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from models.model import fMRICNN

from data_loader.dataloader import fMRICNNcustomDataset

from losses.crossentropy import categorical_cross_entropy

def parse_args():
	parser = argparse.ArgumentParser()

	#model)
	parser.add_argument('--epochs', default=100, type=int, metavar='N',
				help="number of epochs to run")
	parser.add_argument('-b', '--batch_size', default=2, type=int, metavar='N',
				help="mini-batch size (default: 2)")
	parser.add_argument('--early_stopping', default=40, type=int, metavar='N',
				help="early-stopping (default: 40)")
	parser.add_argument('--num_workers', default=6, type=int)
	parser.add_argument('--optimizer', default='Adam', choices=['Adam','SGD'],
				help="Optimizer (default: Adam)")
	parser.add_argument('--lr', default=1e-2, type=float, metavar='LR',
				help="initial learning rate")
	parser.add_argument('--weight_decay', default=1e-4, type=float, 
				help="weight decay")
	
	config = parser.parse_args()

	return config

def train(trainloader, net, criterion, optimizer, train_len, config):
	training_start_time = time.time()
	net.train()
	
	#initialization
	n_classes = 10
	best_dice = 0
	train_losses, val_losses= [], []
	running_loss = 0

	pbar = tqdm(total=(train_len/config['batch_size']), desc='Training')

	for tr_idx, data_samples in (enumerate(trainloader)):
		optimizer.zero_grad()
		volume, labels = data_samples
		volume = volume.cuda()
		labels = labels.long().cuda()
		outputs = net(volume)
		loss = criterion(input_=outputs, target =labels) 

		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		pbar.update(1)
	pbar.close()
	training_loss = running_loss / (2 * tr_idx)
	
	return training_loss

def validate(valloader, net, criterion, config, dataset):
	#validate
	net.eval()

	#initialization
	n_classes = 3
	val_loss = 0

	with torch.no_grad():
		pbar = tqdm(total=(len(dataset)/config['batch_size']),desc='Validation')
		
		for val_idx, data_samples in enumerate(valloader):
			volume, labels = data_samples
			volume = volume.cuda()

			labels = labels.long().cuda()
				
			outputs = net(volume)
			validation_loss_current_model = criterion(input_=outputs, target =labels)
			val_loss += criterion(input_=outputs, target =labels)

			pbar.update(1)
		pbar.close()
	validation_loss = val_loss / (2 * val_idx)
	return validation_loss

def main():
	#Get configuration
	config = vars(parse_args())

	#Make model output directory
	file_name = 'cnn_{}_{}_{}'.format(config['batch_size'],config['optimizer'],config['lr'])

	os.makedirs('model_outputs/{}'.format(file_name),exist_ok=True)

	print("Creating directory called {}".format(file_name))

	print('-' * 20)
	print("Configuration settings as follows:")
	for key in config:
		print('{}: {}'.format(key, config[key]))
	print('-' * 20)

	#Save configuration
	with open('model_outputs/{}/config.yml'.format(file_name), 'w') as f:
		yaml.dump(config, f)

	cudnn.benchmark = True

	#Create model
	print("=> Creating model")
	model = fMRICNN()
	model = model.cuda()

	if torch.cuda.device_count() > 1:
		print("Let's use {} GPUs!".format(torch.cuda.device_count()))
		model = nn.DataParallel(model)

	params = filter(lambda p: p.requires_grad, model.parameters())

	if config['optimizer'] == 'Adam':
		optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
		poly_lr = lambda epoch, max_epochs=config['epochs'], initial_lr=config['lr']: initial_lr * (1 - epoch/max_epochs) ** 0.9
		scheduler = LambdaLR(optimizer, lr_lambda=poly_lr)
	elif config['optimizer'] == 'SGD':
		optimizer = optim.SGD(params, lr=config['lr'], weight_decay=config['weight_decay'])
		poly_lr = lambda epoch, max_epochs=config['epochs'], initial_lr=config['lr']: initial_lr * (1 - epoch/max_epochs) ** 0.9
		scheduler = LambdaLR(optimizer, lr_lambda=poly_lr)
	else:
		raise NotImplementedError

        #Dataset loading
	dataset = fMRICNNcustomDataset("/bigpool/export/users/datasets_faprak2020/BOLD5000/cnn/traindata_cnn_dict")

        #Split data randomly into train and test with 80% training and 20% validation
	train_len=int(len(dataset)*0.8)
	val_len=len(dataset)-train_len
	train_dataset,val_dataset=torch.utils.data.random_split(dataset,(train_len,val_len),generator=torch.Generator().manual_seed(42))
	dl_tr = data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
	dl_val = data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=config['num_workers'], pin_memory=True)
	#metrics
	criterion = categorical_cross_entropy()
	log = pd.DataFrame(index=[], columns=['epoch','lr','loss','train_loss','val_loss'])
	best_val_loss=100
	trigger=0

	for epoch in range(config['epochs']):
		#train for 1 epoch
		train_log = train(dl_tr, model, criterion, optimizer, train_len, config)
		
		#evaluate on validation set
		val_log = validate(dl_val, model, criterion, config, dataset)

		#Update Learning Rate		
		poly_lr(epoch)
		scheduler.step()

		print('Training epoch [{}/{}], Training loss:{:.4f}, Validation loss:{:.4f}'.format(
						epoch + 1,
						config['epochs'],
						train_log,
						val_log))

		tmp = pd.Series([
			epoch,
			config['lr'],
			train_log,
			val_log,
		], index=['epoch','lr','loss','val_loss'])

		log = log.append(tmp, ignore_index=True)
		log.to_csv('model_outputs/{}/log.csv'.format(file_name), index=False)

		trigger += 1

		
		if val_log < best_val_loss:
			if torch.cuda.device_count() > 1:
				torch.save(model.module.state_dict(), 'model_outputs/{}/best_model.pth'.format(file_name))
			else:
				torch.save(model.state_dict(), 'model_outputs/{}/best_model.pth'.format(file_name))
			torch.save(optimizer.state_dict(), 'model_outputs/{}/best_optimizer.pth'.format(file_name))
			best_val_loss = val_log
			print("=> saved best model as validation loss is lower than previous validation loss")
			trigger = 0
                
		#Save snapshot
		if torch.cuda.device_count() > 1:
			torch.save(model.module.state_dict(), 'model_outputs/{}/last_model.pth'.format(file_name))
		else:
			torch.save(model.state_dict(), 'model_outputs/{}/last_model.pth'.format(file_name))
		torch.save(optimizer.state_dict(), 'model_outputs/{}/last_optimizer.pth'.format(file_name))

		#early stopping
		
		if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
			print("=> early stopping")
			break
                
		torch.cuda.empty_cache()

if __name__ == '__main__':
	main()
