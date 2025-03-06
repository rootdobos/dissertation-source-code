import numpy as np
import torch 
import os
class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        return acc, correct, count
        
class EpochLogger():

    def __init__(self,log_dir,n_classes):
        self.log_dir=log_dir
        self.n_classes=n_classes
        os.makedirs(log_dir, exist_ok=True)
        self.train_log=open(os.path.join(log_dir,"train_log.csv"),"w")
        class_list=[f"class{i}" for i in range(n_classes)]
        self.train_log.write(f"epoch,loss,instance_loss,accuracy,macro_f1,weighted_f1,qwk,{','.join(class_list)}\n")
        self.validation_log=open(os.path.join(log_dir,"validation_log.csv"),"w")
        self.validation_log.write(f"epoch,loss,instance_loss,accuracy,macro_f1,weighted_f1,qwk,{','.join(class_list)}\n")
        
    def log_train(self,epoch,losses,metrics,acc_logger):
        class_acc=[]
        for i in range(self.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            class_acc.append(f"{acc}")
        self.train_log.write((f"{epoch},{losses['loss']},{losses['instance_loss']},"
                              f"{metrics['accuracy']},{metrics['macro_f1']},{metrics['weighted_f1']},{metrics['qwk']},"
                              f"{','.join(class_acc)}\n"))
    def log_validate(self,epoch,losses,metrics,acc_logger):
        class_acc=[]
        for i in range(self.n_classes):
            acc, correct, count = acc_logger.get_summary(i)
            class_acc.append(f"{acc}")
        self.validation_log.write((f"{epoch},{losses['loss']},{losses['instance_loss']},"
                              f"{metrics['accuracy']},{metrics['macro_f1']},{metrics['weighted_f1']},{metrics['qwk']},"
                              f"{','.join(class_acc)}\n"))
    def __del__(self):
        self.train_log.close()
        self.validation_log.close()