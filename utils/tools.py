import os
import csv
import json
import copy
import random
import numpy as np
import pandas as pd
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel,BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment
from torch.autograd import Variable

class WeightEMA(object):
    def __init__(self, alpha, model, ema_model):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        # self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.data.copy_(param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def clustering_score(y_true, y_pred):
    return {'ACC': round(clustering_accuracy_score(y_true, y_pred)*100, 2),
            'ARI': round(adjusted_rand_score(y_true, y_pred)*100, 2),
            'NMI': round(normalized_mutual_info_score(y_true, y_pred)*100, 2)}

#https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py#L70
def mask_tokens(inputs, tokenizer,\
    special_tokens_mask=None, mlm_probability=0.15):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs==0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class view_generator:
    def __init__(self, tokenizer, rtr_prob, seed):
        set_seed(seed)
        self.tokenizer = tokenizer
        self.rtr_prob = rtr_prob
    
    def random_token_replace(self, ids):
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        ids, _ = mask_tokens(ids, self.tokenizer, mlm_probability=0.25)
        random_words = torch.randint(len(self.tokenizer), ids.shape, dtype=torch.long)
        indices_replaced = torch.where(ids == mask_id)
        ids[indices_replaced] = random_words[indices_replaced]
        return ids

    def shuffle_tokens(self, ids):
        view_pos = []
        for inp in torch.unbind(ids):
            new_ids = copy.deepcopy(inp)
            special_tokens_mask = self.tokenizer.get_special_tokens_mask(inp, already_has_special_tokens=True)
            sent_tokens_inds = np.where(np.array(special_tokens_mask) == 0)[0]
            inds = np.arange(len(sent_tokens_inds))
            np.random.shuffle(inds)
            shuffled_inds = sent_tokens_inds[inds]
            inp[sent_tokens_inds] = new_ids[shuffled_inds]
            view_pos.append(new_ids)
        view_pos = torch.stack(view_pos, dim=0)
        return view_pos

def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels

def entropy(x, input_as_probabilities=False,q=[]):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = 1e-8)
        q_ = torch.clamp(q,min=1e-8)
        b =  x_ * torch.log(x_/q_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))
    
def linear_rampup(current, start ,end):
    """Linear rampup"""
    assert current >= 0 and end >= 0
    if current <= start:
        return 0
    elif current >= end:
        return 1.0
    else:
        return (current-start) / (end-start)

class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        self.batch_size = len(z_i)
        self.mask = self.mask_correlated_samples(self.batch_size)
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss

import torch

def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)
    
    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)
    
    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)

class FocalLoss(nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='none', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss
    
class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_num =  class_num

    def forward(self, predict, target):
        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, self.class_num) #获取target的one hot编码
        ids = target.view(-1, 1) 
        alpha = self.alpha[ids.data.view(-1)] # 注意，这里的alpha是给定的一个list(tensor#),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
# 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss