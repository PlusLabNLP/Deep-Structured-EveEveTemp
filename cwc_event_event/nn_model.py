import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(123)

class BiLSTM(nn.Module):
    def __init__(self, emb, emb_pos, args):

        super(BiLSTM, self).__init__()
        self.num_words = emb.shape[0]
        self.embed_size = emb.shape[1]
        print(emb.shape)
        self.num_pos_tags = len(args.pos2idx) + 1 # add one for <unk>
        self.hid_size = args.hid
        self.batch_size = args.batch
        self.num_layers = args.num_layers
        self.num_classes = len(args.label_to_id)
        self.num_causal = args.num_causal
        self.out_win_pred = False if args.data_type == "red" else True
        self.none_idx = args.label_to_id['NONE'] if args.data_type == "red" else -1
        self.dropout = args.dropout
        self.attention = args.attention
        
        ### embedding layer
        self.emb = nn.Embedding(self.num_words, self.embed_size, sparse=True)
        self.emb.weight = Parameter(torch.FloatTensor(emb))
        self.emb.weight.requires_grad = False
        self.kernel = (args.kernel, args.kernel)

        ### pos embeddinig -- one-hot vector
        self.emb_pos = nn.Embedding(self.num_pos_tags, self.num_pos_tags, sparse=True)
        self.emb_pos.weight = Parameter(torch.FloatTensor(emb_pos))
        self.emb_pos.weight.requires_grad = False

        ### RNN layer
        self.lstm = nn.LSTM(self.embed_size + self.num_pos_tags, self.hid_size, self.num_layers, bias = False, bidirectional=True)
        
        self.usefeature = args.usefeature
        print('usefeature', self.usefeature)
        if self.usefeature:
            self.linear1 = nn.Linear(self.hid_size*4+args.n_fts, self.hid_size)
        else:
            self.linear1 = nn.Linear(self.hid_size*4, self.hid_size)
        self.linear2 = nn.Linear(self.hid_size, self.num_classes)
        self.linear_c = nn.Linear(self.hid_size, self.num_causal) 
        self.linear_attn = nn.Linear(self.hid_size, self.hid_size)

        self.dropout = nn.Dropout(p=args.dropout)
        self.softmax = nn.Softmax(dim=1)
        self.act = nn.Tanh()
    
    def cal_weights(self, tar_vec, all_vec):
        # tar_vec: batch_size * hid_size
        # all_vec: batch_size * sent_len * hid_size
        # output: weights: batch_size * sent_len                    

        batch_size = all_vec.size()[0]
        sent_len = all_vec.size()[1]
        tar_tensor = tar_vec.unsqueeze(1)
        all_vec = self.linear_attn(all_vec.view(batch_size*sent_len, -1)).view(batch_size, sent_len, -1)

        distance = "cosine_sim"
        if distance == "cosine_sim":
            sims = torch.nn.functional.cosine_similarity(tar_tensor.repeat(1, sent_len, 1), all_vec, dim = 2)
            attns = self.softmax(sims)
        else:
            sims = tar_tensor.bmm(all_vec.transpose(1, 2)).squeeze(1)
            attns = self.softmax(sims)
            # attns.div(torch.sum(attns, dim=1))
        return attns #attns.div(torch.sum(attns, dim=1))

    def attention_weighting(self, ltar_f, ltar_b, rtar_f, rtar_b, out):
        # input: original target hidden vectors
        # output: weighted vectors by attention

        sent_len = out.size()[0]
        batch_size = out.size()[1]

        all_f = out[:, :, :self.hid_size].reshape(batch_size, sent_len, -1)
        all_b = out[:, :, self.hid_size:].reshape(batch_size, sent_len, -1)

        # compute attention weights                                                                                 
        w_ltar_f = self.cal_weights(ltar_f, all_f).unsqueeze(2)
        w_ltar_b = self.cal_weights(ltar_b, all_b).unsqueeze(2)
        w_rtar_f = self.cal_weights(rtar_f, all_f).unsqueeze(2)
        w_rtar_b = self.cal_weights(rtar_b, all_b).unsqueeze(2)

        # compute weighted sum
        ltar_f = torch.sum(w_ltar_f.repeat(1, 1, self.hid_size) * all_f, dim=1)
        ltar_b = torch.sum(w_ltar_b.repeat(1, 1, self.hid_size) * all_b, dim=1)
        rtar_f = torch.sum(w_rtar_f.repeat(1, 1, self.hid_size) * all_f, dim=1)
        rtar_b = torch.sum(w_rtar_b.repeat(1, 1, self.hid_size) * all_b, dim=1)                                                                                  
        return ltar_f, ltar_b, rtar_f, rtar_b

    def forward(self, labels, sent, lidx_start, lidx_end, ridx_start, ridx_end, pred_inds=[], flip = False, causal = False, vat=False):
        use_cuda = self.emb.weight.is_cuda

        batch_size = labels.size()[0]

        ### look up the embedding for sencetences
        # if in VAT training, simply pass in the noisy input
        if vat:
            emb = sent[0]
        else:
            emb = self.dropout(self.emb(sent[0]))

        ### create embeddings for pos tags
        pos = self.emb_pos(sent[1])

        ### obtain hidden vars based on start and end idx 
        out, _ = self.lstm(torch.cat((emb, pos), dim=2))
        
        ### flatten hidden vars into a long vector
        ltar_f = out[lidx_end, :, :self.hid_size].view(batch_size, -1)
        ltar_b = out[lidx_start, :, self.hid_size:].view(batch_size, -1)
        rtar_f = out[ridx_end, :, :self.hid_size].view(batch_size, -1)
        rtar_b = out[ridx_start, :, self.hid_size:].view(batch_size, -1)

        if self.attention:
            ltar_f, ltar_b, rtar_f, rtar_b = self.attention_weighting(ltar_f, ltar_b, rtar_f, rtar_b, out)

        if flip:
            tar = self.dropout(torch.cat((rtar_f, rtar_b, ltar_f, ltar_b), dim=1))
            #tar = self.dropout(torch.cat((rtar_b, rtar_f, ltar_b, ltar_f), dim=1))
        else:
            tar = self.dropout(torch.cat((ltar_f, ltar_b, rtar_f, rtar_b), dim=1))

        if self.usefeature:
            out = torch.cat((tar, sent[2]), dim=1)
        else:
            out = tar
        # linear prediction
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout(out)
        # causal relation 
        if causal:
            #out = Variable(out.data, requires_grad=False)
            out = self.linear_c(out)
        else:
            out = self.linear2(out)
        prob = self.softmax(out)
        return out, prob