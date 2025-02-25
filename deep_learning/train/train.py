import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from deep_learning.loaders.feature_dataset import FeatureDataset

from torch import nn
from torchinfo import summary
import torch.optim as optim

from deep_learning.train.loggers import Accuracy_Logger, EarlyStopping

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model,df, feature_path,epochs):
	train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)

	train_dataset=FeatureDataset(train_df,feature_path)
	val_dataset=FeatureDataset(validation_df,feature_path)

	train_loader=DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=8)
	val_loader=DataLoader(val_dataset, batch_size=1, shuffle=True,num_workers=8)

	loss_fn=nn.CrossEntropyLoss()
	instance_loss_fn=nn.CrossEntropyLoss()

	_= model.to(device)

	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-5)
	early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)
	
	for epoch in range(epochs):
		train_loop(epoch, model, train_loader, optimizer, 6, 0.5,loss_fn)
		validate_clam(model, val_loader, 6, loss_fn)
		
	
def train_loop(epoch,model,loader,optimizer,n_classes,bag_weight,loss_fn):
	model.train()
	acc_logger=Accuracy_Logger(n_classes=n_classes)
	inst_logger=Accuracy_Logger(n_classes=n_classes)
	train_loss, train_accuracy,train_inst_loss,inst_count=0. , 0. , 0. , 0
	confusion_matrix=np.zeros((n_classes,n_classes))
	print('\n')
	for batch_idx,(data,label) in enumerate(loader):
		data,label=data.to(device).squeeze(),label.to(device)
		logits, Y_prob,Y_hat,_,instance_dict=model(data,label=label,instance_eval=True)

		acc_logger.log(Y_hat,label)
		loss=loss_fn(logits,label)
		loss_value=loss.item()


		instance_loss=instance_dict['instance_loss']
		inst_count+=1
		instance_loss_value=instance_loss.item()
		train_inst_loss+=instance_loss_value

		total_loss=bag_weight*loss +(1-bag_weight)*instance_loss
		
		inst_preds=instance_dict['inst_preds']
		inst_labels=instance_dict['inst_labels']
		inst_logger.log_batch(inst_preds,inst_labels)

		train_loss+=loss_value
		if(batch_idx+1)%20==0:
			print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
				'label: {}, bag_size: {}'.format(label.item(), data.size(0)))
		accuracy=calculate_equal_predictions(Y_hat,label)
		confusion_matrix[label][Y_hat]+=1
		train_accuracy += accuracy
		total_loss.backward()
		optimizer.step()
		optimizer.zero_grad()

	train_loss /= len(loader)
	train_accuracy /= len(loader)
	train_inst_loss/=len(loader)

	print('Epoch {} Summary: train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_accuracy: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_accuracy))
	print("Slide Accuracy:")
	for i in range(n_classes):
		acc, correct, count = acc_logger.get_summary(i)
		print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
	print("Patch Accuracy:")
	for i in range(n_classes):
		acc, correct, count = inst_logger.get_summary(i)
		print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
	print(confusion_matrix)

def validate_clam( model, loader, n_classes, loss_fn = None):
	model.eval()
	acc_logger = Accuracy_Logger(n_classes=n_classes)
	inst_logger = Accuracy_Logger(n_classes=n_classes)
	val_loss,val_accuracy,val_inst_loss,inst_count= 0. , 0. , 0. , 0
	confusion_matrix=np.zeros((n_classes,n_classes))
	prob = np.zeros((len(loader), n_classes))
	labels = np.zeros(len(loader))
	sample_size = model.k_sample
	with torch.inference_mode():
		for batch_idx, (data, label) in enumerate(loader):
			data, label = data.to(device).squeeze(), label.to(device)      
			logits, Y_prob, Y_hat, _, instance_dict = model(data, label=label, instance_eval=True)
			acc_logger.log(Y_hat, label)
			
			loss = loss_fn(logits, label)

			val_loss += loss.item()

			instance_loss = instance_dict['instance_loss']
			
			inst_count+=1
			instance_loss_value = instance_loss.item()
			val_inst_loss += instance_loss_value

			inst_preds = instance_dict['inst_preds']
			inst_labels = instance_dict['inst_labels']
			inst_logger.log_batch(inst_preds, inst_labels)

			prob[batch_idx] = Y_prob.cpu().numpy()
			labels[batch_idx] = label.item()
			
			accuracy = calculate_equal_predictions(Y_hat, label)
			val_accuracy += accuracy
			confusion_matrix[label][Y_hat]+=1

	val_accuracy /= len(loader)
	val_loss /= len(loader)

	print('\nVal Set, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(val_loss, val_accuracy))
	for i in range(n_classes):
		acc, correct, count = acc_logger.get_summary(i)
		print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
	print(confusion_matrix)
	 


def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def calculate_equal_predictions(Y_hat, Y):
	"""classification error rate"""
	return Y_hat.float().eq(Y.float()).float().mean().item()


