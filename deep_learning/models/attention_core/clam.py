from torch import nn
import torch.nn.functional as F
import torch
from .attention import Attn_Net_Gated
import numpy as np


class Clam(nn.Module):
    def __init__(self, feature_vector_length=1024, dropout=0., k_sample=8, n_classes=2,
                 loss_fn=nn.CrossEntropyLoss(), subtyping=False,):
        super(Clam, self).__init__()
        size = [feature_vector_length, 512, 256]

        fc = [nn.Linear(size[0], size[1]),
              nn.ReLU(),
              nn.Dropout(dropout),
              Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)]
        self.attention_net = nn.Sequential(*fc)

        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)

        instance_classifiers = [nn.Linear(size[1], 2)for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)

        self.k_sample = k_sample
        self.instance_loss_fn = loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

    def common_forward_part_before_logits(self, h, label, instance_eval, attention_only):
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)
        total_inst_loss = 0.0
        all_preds = []
        all_targets = []
        if instance_eval:

            inst_labels = F.one_hot(
                label, num_classes=self.n_classes).squeeze()
            for i in range(self.n_classes):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:
                    instance_loss, preds, targets = self.inst_eval(
                        A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(
                            A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss
            if self.subtyping:
                total_inst_loss /= self.n_classes
        M = torch.mm(A, h)
        return M, A_raw,  total_inst_loss, all_targets, all_preds

    def common_forward_part_after_logits(self, logits, M, A_raw, instance_eval, return_features, total_inst_loss, all_targets, all_preds):
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, Y_prob, Y_hat, A_raw, results_dict

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        if attention_only:
            return self.common_forward_part_before_logits(h, label, instance_eval, attention_only)
        else:
            M, A_raw, total_inst_loss, all_targets, all_preds = self.common_forward_part_before_logits(
                h, label, instance_eval, attention_only)

            logits = torch.empty(1, self.n_classes).float().to(M.device)
            for c in range(self.n_classes):
                logits[0, c] = self.classifiers[c](M[c])

            return self.common_forward_part_after_logits(logits, M, A_raw, instance_eval, return_features, total_inst_loss, all_targets, all_preds)

    # instance-level evaluation for in-the-class attention branch

    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k_sample = min(self.k_sample, A.numel())
        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k_sample, device)
        n_targets = self.create_negative_targets(k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        k_sample = min(self.k_sample, A.numel())
        top_p_ids = torch.topk(A, k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()


class ClamSimple(Clam):
    def __init__(self, feature_vector_length=1024, dropout=0., k_sample=8, n_classes=2,
                 loss_fn=nn.CrossEntropyLoss(), subtyping=False,):
        super().__init__(feature_vector_length, dropout, k_sample, n_classes,
                         loss_fn, subtyping)

        size = [feature_vector_length, 512, 256]

        self.classifiers = nn.Linear(size[1], 1)

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        if attention_only:
            return self.common_forward_part_before_logits(h, label, instance_eval, attention_only)
        else:
            M, A_raw, total_inst_loss, all_targets, all_preds = self.common_forward_part_before_logits(
                h, label, instance_eval, attention_only)

            logits = self.classifiers(M)

            logits = logits.reshape(1, 6)
            return self.common_forward_part_after_logits(logits, M, A_raw, instance_eval, return_features, total_inst_loss, all_targets, all_preds)
