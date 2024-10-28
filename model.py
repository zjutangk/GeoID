"""
modified from
https://github.com/thuiar/DeepAligned-Clustering/blob/main/model.py
"""

from utils.tools import *
from utils.contrastive import SupConLoss
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
        
class BertForModel(nn.Module): #######################33pretrain
    def __init__(self,model_name, num_labels, device=None):
        super(BertForModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.device = device
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.classifier = nn.Linear(768, self.num_labels)
        # self.classifier = ETF_Classifier(768, num_classes=self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.backbone.to(self.device)
        self.classifier.to(self.device)

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        """logits are not normalized by softmax in forward function"""
        outputs = self.backbone(**X, output_hidden_states=True)
        # extract last layer [CLS]
        CLSEmbedding = outputs.hidden_states[-1][:,0]
        CLSEmbedding = self.dropout(CLSEmbedding)
        logits = self.classifier(CLSEmbedding)
        output_dir = {"logits": logits}
        if output_hidden_states:
            output_dir["hidden_states"] = outputs.hidden_states[-1][:, 0]
        if output_attentions:
            output_dir["attentions"] = outputs.attention
        return output_dir

    def mlmForward(self, X, Y):
        outputs = self.backbone(**X, labels=Y)
        return outputs.loss

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output
    
    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)


class CLBert(nn.Module):
    def __init__(self,model_name, device, label_num,feat_dim=128):
        super(CLBert, self).__init__()
        self.model_name = model_name
        self.device = device
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, feat_dim)
        )
        self.classifier_new = ETF_Classifier(feat_in=feat_dim, num_classes=label_num) ##self.num_labels
        self.backbone.to(self.device)
        self.head.to(device)
        self.classifier_new.to(device)
        
    def forward(self, X, output_hidden_states=True, output_attentions=False, output_logits=True):
        """logits are not normalized by softmax in forward function"""
        outputs = self.backbone(**X, output_hidden_states=True, output_attentions=True)
        cls_embed = outputs.hidden_states[-1][:,0]
        pooled_output=self.head(cls_embed)
        cur_M = self.classifier_new.ori_M.cuda()
        feat_norm = self.classifier_new(pooled_output)
        output_dir = {"features": feat_norm}
        logit = torch.matmul(feat_norm, cur_M)
        # output_dir={"logits":logit}

        if output_hidden_states:
            output_dir["hidden_states"] = cls_embed
        if output_attentions:
            output_dir["attentions"] = outputs.attentions
        if output_logits:
            output_dir["logits"]=logit
        # print(logit.shape)
        # print(output_dir["hidden_states"].shape)
        return output_dir

    def loss_cl(self, embds, label=None, mask=None, temperature=0.07, base_temperature=0.07):
        """compute contrastive loss"""
        loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        output = loss(embds, labels=label, mask=mask)
        return output
    
    def save_backbone(self, save_path):
        self.backbone.save_pretrained(save_path)

class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False, try_assert=True):
        super(ETF_Classifier, self).__init__()
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes,try_assert)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * torch.matmul(P, I-((1/num_classes) * one))
        self.ori_M = M.cuda()
        self.ori_M.requires_grad_(False)
        #import ipdb;ipdb.set_trace()
        self.LWS = LWS
        self.reg_ETF = reg_ETF
#        if LWS:
#            self.learned_norm = nn.Parameter(torch.ones(1, num_classes))
#            self.alpha = nn.Parameter(1e-3 * torch.randn(1, num_classes).cuda())
#            self.learned_norm = (F.softmax(self.alpha, dim=-1) * num_classes)
#        else:
#            self.learned_norm = torch.ones(1, num_classes).cuda()

        self.BN_H = nn.BatchNorm1d(feat_in)
        if fix_bn:
            self.BN_H.weight.requires_grad = False
            self.BN_H.bias.requires_grad = False

    def generate_random_orthogonal_matrix(self, feat_in, num_classes,try_assert):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        if try_assert:
            assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        x = self.BN_H(x)
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        return x