import torch
import os
class EarlyStopping:
    def __init__(self,save_path, patience=5,min_delta=0.0):
        self.save_path=save_path
        self.min_delta=min_delta
        self.patience=patience
        self.best_loss=float('inf')
        self.counter=0

    def __call__(self,epoch, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(),os.path.join(self.save_path,f"model_weights_{epoch}.pth"))
            print(f"Validation loss improved. Saving model to {self.save_path}")
        else:
            self.counter += 1
            print(f"!!!EarlyStopping counter: {self.counter}/{self.patience} | Best loss: {self.best_loss}")
            if self.counter >= self.patience:
                print("!!!Early stopping triggered!!!")
                return True 
        return False 