import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import Parameter

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(123)
torch.cuda.manual_seed(0)

class BiLSTM(nn.Module):
    def __init__(self, emb, emb_pos, args):
        super(BiLSTM, self).__init__()
        self.num_words = emb.shape[0]
        self.embed_size = emb.shape[1]
        self.num_pos_tags = len(args.pos2idx) + 2  # add one for <unk>, one for <pad>
        self.hid_size = args.hid
        self.num_layers = args.num_layers
        self.num_classes = len(args.label_to_id)
        self.num_causal = args.num_causal
        #self.out_win_pred = False if args.data_type == "red" else True # not sure what it is
        #self.none_idx = args.label_to_id['NONE'] if args.data_type == "red" else -1
        self.dropout = args.dropout
        self.attention = args.attention
        self.bert = args.bert_fts
        self.sparse=args.sparse_emb
        
        ### embedding layer
        if self.bert:
            self.embed_size = args.bert_dim
        else:
            self.emb = nn.Embedding(self.num_words, self.embed_size, padding_idx=0, sparse=self.sparse)
            self.emb.weight = Parameter(torch.FloatTensor(emb))
            self.emb.weight.requires_grad = False
        
        ### pos embeddinig -- one-hot vector
        self.emb_pos = nn.Embedding(self.num_pos_tags, self.num_pos_tags, padding_idx=37, sparse=self.sparse)
        self.emb_pos.weight = Parameter(torch.FloatTensor(emb_pos))
        self.emb_pos.weight.requires_grad = args.train_pos_emb

        ### RNN layer
        #self.lstm = nn.LSTM(self.embed_size+self.num_pos_tags,self.hid_size,
        #                    self.num_layers,bias = False,bidirectional=True)
        self.lstm = nn.LSTM(self.embed_size+self.num_pos_tags, self.hid_size,
                            self.num_layers, bidirectional=True, batch_first=True)
        
        self.usefeature = args.usefeature
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
        return attns

    def attention_weighting(self, ltar_f, ltar_b, rtar_f, rtar_b, out):
        # input: original target hidden vectors
        # output: weighted vectors by attention
        sent_len = out.size()[1]
        batch_size = out.size()[0]

        all_f = out[:, :, :self.hid_size]
        all_b = out[:, :, self.hid_size:]
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

    def forward(self, seq_lens, sent, lidx_start, lidx_end, ridx_start, ridx_end, 
                pred_inds=[], flip = False, causal = False, vat=False):
        '''
        sent[0]: the input sentence (represent in index) in shape (batch_size, seq_len)
        sent[1]: the input POS tage (represent in index) in shape (batch_size, seq_len)
        sent[2]: the linguistic features in shape (batch, n_fts)
        seq_lens: the sequence length for each batch e.g.: [8,4,3,...]
        idx_start/end: a batch of index that assign which element to take. [5, 3, 2, 8,...] 
        '''

        # look up the embedding for sencetences
        # if in VAT training, simply pass in the noisy input
        if vat:
            emb = sent[0]
        elif self.bert:
            emb = self.dropout(sent[0])
        else:
            emb = self.dropout(self.emb(sent[0]))
        
        # create embeddings for pos tags
        pos = self.emb_pos(sent[1])
        # pack and pass to lstm module and then pad again
        inputs = torch.cat((emb, pos), dim=2)
        pack_inputs = pack_padded_sequence(inputs, seq_lens, batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(pack_inputs)
        out, seq_lens = pad_packed_sequence(out, batch_first=True, padding_value=0.0) # (batch, seq_len, 2*hid_size)
        
        ### obtain hidden vars based on start and end idx 
        batch_size = len(seq_lens)
        lidx_e_idx = lidx_end.unsqueeze(1).expand((out.size(0), out.size(2))).unsqueeze(1) # (batch, 1, 2*hid_size)
        ltar_f = torch.gather(out, dim=1, index=lidx_e_idx).squeeze(1)[:,self.hid_size:]
        lidx_s_idx = lidx_start.unsqueeze(1).expand((out.size(0), out.size(2))).unsqueeze(1)
        ltar_b = torch.gather(out, dim=1, index=lidx_s_idx).squeeze(1)[:,:self.hid_size]
        ridx_e_idx = ridx_end.unsqueeze(1).expand((out.size(0), out.size(2))).unsqueeze(1)
        rtar_f = torch.gather(out, dim=1, index=ridx_e_idx).squeeze(1)[:,self.hid_size:]
        ridx_s_idx = ridx_start.unsqueeze(1).expand((out.size(0), out.size(2))).unsqueeze(1)
        rtar_b = torch.gather(out, dim=1, index=ridx_s_idx).squeeze(1)[:,:self.hid_size]
        
        if self.attention:
            ltar_f, ltar_b, rtar_f, rtar_b = self.attention_weighting(ltar_f, ltar_b, rtar_f, rtar_b, out)

        if flip:
            tar = self.dropout(torch.cat((rtar_f, rtar_b, ltar_f, ltar_b), dim=1))
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
            out = self.linear_c(out)
        else:
            out = self.linear2(out)
        prob = self.softmax(out) # batch x num_labels
        return out, prob
