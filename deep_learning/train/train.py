import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from deep_learning.loaders.feature_dataset import FeatureDataset
from deep_learning.utils.metrics import quadratic_weighted_kappa_cf,compute_precision_recall_f1
from torch import nn
import torch.optim as optim
import os
from deep_learning.train.loggers import Accuracy_Logger, EpochLogger
import datetime
from queue import Queue

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model,df, feature_path,epochs,patience,log_dir):
	now = datetime.datetime.now()
	log_dir=os.path.join(log_dir,now.strftime("%Y%m%d_%H%M%S"))
	train_df, validation_df = train_test_split(df, test_size=0.2, random_state=42)

	train_dataset=FeatureDataset(train_df,feature_path)
	val_dataset=FeatureDataset(validation_df,feature_path)

	train_loader=DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=0)
	val_loader=DataLoader(val_dataset, batch_size=1, shuffle=True,num_workers=0)

	loss_fn=nn.CrossEntropyLoss()
	instance_loss_fn=nn.CrossEntropyLoss()

	_= model.to(device)

	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-5)
	logger=EpochLogger(log_dir,6)
	
	last_k=Queue()
	count_p=0
	for epoch in range(epochs):
		t_losses, t_metrics,t_confusion_matrix, t_acc_logger=train_loop(epoch, model, train_loader, optimizer, 6, 0.5,loss_fn)
		v_losses, v_metrics,v_confusion_matrix, v_acc_logger=validate_clam(model, val_loader, 6, loss_fn)
		logger.log_train(epoch,t_losses,t_metrics,t_acc_logger)
		logger.log_validate(epoch,v_losses,v_metrics,v_acc_logger)
		if epoch>0:
			if min(list(last_k.queue))>=t_losses['loss']:
				count_p=0
			else:
				count_p+=1
				print(f"Loss not improved since {count_p} epoch")
				if count_p>patience:
					print(f"Early Stop!")
					break
		last_k.put(t_losses['loss'])
		torch.save(model.state_dict(),os.path.join(log_dir,f"model_weights_{epoch}.pth"))


	#torch.save(model.state_dict(),os.path.join(log_dir,"model_weights.pth"))

	del logger

	
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
		if(batch_idx+1)%500==0:
			print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()))
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
	_,_,_,macro_f1,weighted_f1=compute_precision_recall_f1(confusion_matrix)
	print(f"Macro-F1: {macro_f1:.4f}; Weighted F1:{weighted_f1:.4f}")
	print("Slide Accuracy:")
	for i in range(n_classes):
		acc, correct, count = acc_logger.get_summary(i)
		print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
	print(confusion_matrix)
	qwk_score=quadratic_weighted_kappa_cf(confusion_matrix,n_classes)
	print(f"Quadratic Weighted Kappa: {qwk_score:.4f}")
	losses={
		"loss": train_loss,
		"instance_loss":train_inst_loss
	}
	metrics={
		"accuracy":train_accuracy,
		"macro_f1": macro_f1,
		"weighted_f1":weighted_f1,
		"qwk":qwk_score
	}
	return losses, metrics,confusion_matrix, acc_logger

def validate_clam( model, loader, n_classes, loss_fn = None):
	torch.cuda.empty_cache()
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
	_,_,_,macro_f1,weighted_f1=compute_precision_recall_f1(confusion_matrix)
	print(f"Macro-F1: {macro_f1:.4f}; Weighted F1:{weighted_f1:.4f}")
	for i in range(n_classes):
		acc, correct, count = acc_logger.get_summary(i)
		print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
	print(confusion_matrix)
	qwk_score=quadratic_weighted_kappa_cf(confusion_matrix,n_classes)
	print(f"Quadratic Weighted Kappa: {qwk_score:.4f}")
	losses={
		"loss": val_loss,
		"instance_loss":val_inst_loss
	}
	metrics={
		"accuracy":val_accuracy,
		"macro_f1": macro_f1,
		"weighted_f1":weighted_f1,
		"qwk":qwk_score
	}
	return losses, metrics,confusion_matrix, acc_logger
	 


def get_optim(model, args):
	if args.opt == "adam":
		optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
	else:
		raise NotImplementedError
	return optimizer

def calculate_equal_predictions(Y_hat, Y):
	return Y_hat.float().eq(Y.float()).float().mean().item()


