import torch
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import pandas as pd
import os 

def get_attention_scores(model, features,k=1, to_device=True):
    if to_device:
        _= model.to(device)
        model.eval()
    features=features.to(device).squeeze()
    with torch.inference_mode():
        logits, Y_prob, Y_hat, A, _ = model(features)
        Y_hat= Y_hat.item()
        A=A.cpu().numpy()
        #print('Y_hat: {}, Y_prob: {}'.format(Y_hat,  ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))
        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()

    return Y_prob,A

def get_best_attention_scores(model, features):
    Y_prob,A=get_attention_scores(model,features)
    print(Y_prob)
    #print(A.shape)
    #print(A)
    best_scores=[{"value":np.max(A[:,i]), 
                 "predicted_label":np.argmax(A[:,i])} for i in range(A.shape[1])]
    return best_scores
    #print(best_scores)

def get_tiles_coords(path):
    files = next(os.walk(path))
    images = files[2]
    coords = []
    for image_path in images:
        coord = image_path.split('.')[0].split('_')
        coords.append({
            "x": int(coord[0]),
            "y": int(coord[1])

        })
    return coords


def pair_coords_and_attention_scores(scores, coords):
    return [{**score, **coord} for score, coord in zip(scores, coords)]


def get_min_max_coordinates(data):
    df = pd.DataFrame(data)
    return {
        'min_x': df['x'].min(),
        'max_x': df['x'].max(),
        'min_y': df['y'].min(),
        'max_y': df['y'].min()
    }

def get_statistics_from_attention_scores(A):
    result={}
    for i in range(A.shape[0]):
        result[i]={
            'min':float(A[i,:].min()),
            'max':float(A[i,:].max()),
            'mean':float(A[i,:].mean()),
            'std':float(A[i,:].std())
        }
    return result

def print_attention_scores_statistics(stats):
    for k,v in stats.items():
        print(f"{k} Attention scores summary:")
        print(f"\tMin: \t{v['min']}")
        print(f"\tMax: \t{v['max']}")
        print(f"\tMean: \t{v['mean']}")
        print(f"\tStd: \t{v['std']}")