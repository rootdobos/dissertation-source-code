import torch
import numpy as np
import pandas as pd
import os

from ..models.attention_core.clam import ClamSimple
from .file_and_data_service import FileDataService

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionScoreService():
    @staticmethod
    def get_attention_scores_from_features(model_name, model_weights_path, features_path):

        model = ClamSimple(
            feature_vector_length=1280,
            dropout=0.1,
            k_sample=8,
            n_classes=6,
            subtyping=False
        )
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_weights_path))
        else:
            model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

        features = FileDataService.load_slide_features(features_path)

        Y_prob, scores = get_attention_scores_for_classes(model, features)
        return Y_prob, scores


def get_attention_scores(model, features, k=1, to_device=True):
    if to_device:
        _ = model.to(device)
        model.eval()
    features = features.to(device).squeeze()
    with torch.inference_mode():
        logits, Y_prob, Y_hat, A, _ = model(features)
        Y_hat = Y_hat.item()
        A = A.cpu().numpy()
        # print('Y_hat: {}, Y_prob: {}'.format(Y_hat,  ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))
        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()

    return Y_prob, A


def get_best_attention_scores(model, features):
    Y_prob, A = get_attention_scores(model, features)
    print(Y_prob)

    best_scores = [{"value": np.max(A[:, i]),
                    "predicted_label": np.argmax(A[:, i])} for i in range(A.shape[1])]
    return best_scores


def get_attention_scores_for_classes(model, features):
    Y_prob, A = get_attention_scores(model, features, 1)
    stats = get_attention_scores_for_each_class(A)

    return Y_prob, stats


def pair_coords_and_attention_scores(scores, coords):
    return {"coords": coords, **scores}


def get_statistics_from_attention_scores(A):
    result = {}
    for i in range(A.shape[0]):
        result[i] = {
            'min': float(A[i, :].min()),
            'max': float(A[i, :].max()),
            'mean': float(A[i, :].mean()),
            'std': float(A[i, :].std())
        }
    return result


def get_attention_scores_for_each_class(A):
    result = {}
    for i in range(A.shape[0]):
        result[i] = A[i, :].tolist()
    return result


def print_attention_scores_statistics(stats):
    for k, v in stats.items():
        print(f"{k} Attention scores summary:")
        print(f"\tMin: \t{v['min']}")
        print(f"\tMax: \t{v['max']}")
        print(f"\tMean: \t{v['mean']}")
        print(f"\tStd: \t{v['std']}")


def get_min_max_with_percentile(values, lower_percentile=5, upper_percentile=95):
    result = {}
    for i in range(6):
        intervals_for_bag_class = {}
        for k in range(6):
            data = values[i][k]
            interval = {
                "min": np.percentile(data, lower_percentile),
                "max": np.percentile(data, upper_percentile),
                "median": np.median(data),
            }
            intervals_for_bag_class[k] = interval
        result[i] = intervals_for_bag_class
    return result


def aggregate_labels(statlist):
    stats = {}
    for i in range(6):
        labeled = list(filter(lambda x: x['label'] == i, statlist))
        length = len(labeled)
        result_dict = {j: [] for j in range(6)}
        for index, data in enumerate(labeled):
            for k in range(6):
                result_dict[k].extend(data[k])
        stats[i] = result_dict
    return stats
