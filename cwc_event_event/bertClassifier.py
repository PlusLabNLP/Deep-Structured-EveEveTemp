from pathlib import Path
import pickle
import sys
import argparse
from flexnlp import Document
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations
from typing import Iterator, List, Mapping, Union, Optional, Set
import logging as log
import abc
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import math
import time
import copy
from baseline import FlatRelation, print_annotation_stats, print_flat_relation_stats, read_relations, all_red_labels, REDEveEveRelModel, ClassificationReport, matres_label_map, tbd_label_map, new_label_map, red_label_map, rev_map, causal_label_map
from featureFuncs import *
#from utils import *
import multiprocessing as mp
from functools import partial
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from ldctcr import NewDoc, NewRelation, NewEntity
from ldctbd import TBDDoc, TBDRelation, TBDEntity
from pytorch_pretrained_bert.modeling import BertModel, BertConfig, PreTrainedBertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

torch.backends.cudnn.deterministic = True
torch.manual_seed(123)


class BertClassifier(PreTrainedBertModel):

    def __init__(self, config, args):
        super(BertClassifier, self).__init__(config)
        self.hid_size = args.hid
        self.batch_size = args.batch
        self.num_layers = args.num_layers
        self.num_classes = len(args.label_to_id)

        # load pre-trained bert model
        self.bert = BertModel.from_pretrained('bert-base-uncased') #BertModel(config)
        self.dropout = nn.Dropout(p=args.dropout)
        self.lstm = nn.LSTM(config.hidden_size, self.hid_size, self.num_layers, bias = False, bidirectional=True)
        self.linear1 = nn.Linear(self.hid_size*4+args.n_fts, self.hid_size)
        self.linear2 = nn.Linear(self.hid_size, self.num_classes)
        self.act = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.apply(self.init_bert_weights)

    def forward(self, labels, sents, lidx_start, lidx_end, ridx_start, ridx_end, 
                flip=False, causal=False, token_type_ids=None, attention_mask=None):

        batch_size = labels.size()[0]

        out, _ = self.bert(sents[0], token_type_ids, attention_mask, output_all_encoded_layers=False)
        
        #print(sents[0].size())
        #print(out.size())
        #print(sents[2].size())

        #left = torch.mean(out[lidx_start:lidx_end+1, :, :], dim=0)
        #right = torch.mean(out[ridx_start:ridx_end+1, :, :], dim=0)


        out, _ = self.lstm(self.dropout(out))

         ### flatten hidden vars into a long vector                                                                        
        ltar_f = out[lidx_end, :, :self.hid_size].view(batch_size, -1)
        ltar_b = out[lidx_start, :, self.hid_size:].view(batch_size, -1)
        rtar_f = out[ridx_end, :, :self.hid_size].view(batch_size, -1)
        rtar_b = out[ridx_start, :, self.hid_size:].view(batch_size, -1)
        #left = out[lidx_start:lidx_start+1, :, :]
        #right = out[ridx_start:ridx_start+1, :, :]

        #        print(left.size())
        #        print(right.size())
        
        
        #        print(left.size())
        #        print(right.size())

        #out = self.dropout(torch.cat((left, right), dim=1))
        #        print(out.size())

        out = self.dropout(torch.cat((ltar_f, ltar_b, rtar_f, rtar_b), dim=1))
        out = torch.cat((out, sents[2]), dim=1)

        # linear prediction                                                                                      
        out = self.linear1(out)
        out = self.act(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        prob = self.softmax(out)
#        print(kill)
        return out, prob


@dataclass()
class NNClassifier(REDEveEveRelModel):
    """                                                                                                                            
    A simple baseline model which assigns a random class label to each event-event pair                                            
    """
    label_probs: Optional[List[float]] = None

    def create_features(self, ex, args):

        pos2idx = args.pos2idx
        pos_dict = ex.doc.pos_dict #create_pos_dict(ex.doc.nlp_ann.pos_tags())
        all_keys, lidx_start, lidx_end, ridx_start, ridx_end = token_idx(ex.left.span, ex.right.span, pos_dict)
        
        # create an indicator of whether two events are outside of pred_window
        left_mid = (lidx_start + lidx_end) / 2.0
        right_mid = (ridx_start + ridx_end) / 2.0
        pred_ind = False if np.abs(left_mid - right_mid) > args.pred_win else True

        # compute the exact number of left and right neighbors                                                                                                                                      
        if lidx_start != lidx_end:
            lngbrs = args.ngbrs - math.floor((lidx_end - lidx_start) / 2)
            rngbrs = args.ngbrs - math.ceil((lidx_end - lidx_start) / 2)
        else:
            lngbrs = args.ngbrs
            rngbrs = args.ngbrs

        end_pad = rngbrs
        # left_ngbrs out of range, pad left                                                                                                                                                                
        if lidx_start - lngbrs < 0:
            lsent_pos = [('<pad>', '<unk>') for x in range(lngbrs - lidx_start)] + [pos_dict[x] for x in all_keys[:lidx_end+1+rngbrs]]
        # right_nbrs out of range pad right                                                                                                                                                                
        elif lidx_end + rngbrs > len(pos_dict) - 1:
            lsent_pos = [pos_dict[x] for x in all_keys[lidx_start - lngbrs:]] + [('<pad>', '<unk>') for x in range(rngbrs + lidx_end - (len(pos_dict) - 1))]
        # regular cases                                                                                                                                                                                    
        else:                                                                                                                                                                               
            lsent_pos = [pos_dict[x] for x in all_keys[lidx_start - lngbrs : lidx_end + 1 + rngbrs]]

        # adjust target token index in left sentence                                                                                                                                                       
        lidx_end_s = (lidx_end - lidx_start) + lngbrs
        lidx_start_s = lngbrs

        assert lidx_start_s >= 0
        assert lidx_end_s < args.ngbrs * 2
        
        # need to figure out exact number of left and right neighbors                                                                                                                                      
        if ridx_start != ridx_end:
            lngbrs = args.ngbrs - math.floor((ridx_end - ridx_start) / 2)
            rngbrs = args.ngbrs - math.ceil((ridx_end - ridx_start) / 2)
        else:
            lngbrs = args.ngbrs
            rngbrs = args.ngbrs

        end_pad += lngbrs
        
        end_pad -= (np.abs(lidx_end - ridx_start) - 1)
        
        # left_ngbrs out of range, pad left                                                                                                                                                                
        if ridx_start - lngbrs < 0:
            rsent_pos = [('<pad>', '<unk>') for x in range(lngbrs - ridx_start)] + [pos_dict[x] for x in all_keys[:ridx_end+1+rngbrs]]
        # right_nbrs out of range pad right                                                                                                                                                                
        elif ridx_end + rngbrs > len(pos_dict) - 1:
            rsent_pos = [pos_dict[x] for x in all_keys[ridx_start - lngbrs:]] + [('<pad>', '<unk>') for x in range(rngbrs + ridx_end - (len(pos_dict) - 1))]
        # regular cases                                                                                                                                                                                    
        else:
            rsent_pos = [pos_dict[x] for x in all_keys[ridx_start - lngbrs : ridx_end + 1 + rngbrs]]

        assert len(lsent_pos) == 2 * args.ngbrs + 1
        assert len(rsent_pos) == 2 * args.ngbrs + 1

        # adjust target token index in right sentence                                                                                                                                                      
        ridx_end_s = (ridx_end - ridx_start) + lngbrs
        ridx_start_s = lngbrs

        assert ridx_start_s >= 0
        assert ridx_end_s < args.ngbrs * 2

        if lidx_end_s + (ridx_start - lidx_end) < len(lsent_pos):
            lsent_pos = lsent_pos[:lidx_end_s + (ridx_start - lidx_end)]
            rsent_pos = rsent_pos[ridx_start_s:]            
            end_pad = ['<pad>' for _ in range(end_pad)]

            ridx_start_s = lidx_end_s + (ridx_start - lidx_end)
            ridx_end_s = lidx_end_s + (ridx_end - lidx_end)
        else:
           end_pad = []
           ridx_start_s += len(lsent_pos)
           ridx_end_s += len(lsent_pos)

        # lookup token idx                                                                                                                                                                               
        lsent = [x[0].lower() for x in lsent_pos]
        lsent = [args.w2i[t] if t in args.w2i.keys() else 1 for t in lsent]

        rsent = [x[0].lower() for x in rsent_pos]
        rsent = [args.w2i[t] if t in args.w2i.keys() else 1 for t in rsent]

        psent = [args.w2i[p] for p in end_pad]

        # lookup pos_tag idx                                                                                                                                  
        lpos = [pos2idx[p] if p in pos2idx.keys() else len(pos2idx) for _, p in lsent_pos]
        rpos = [pos2idx[p] if p in pos2idx.keys() else len(pos2idx) for _, p in rsent_pos]
        ppos = [len(pos2idx) for p in end_pad]
        
        assert len(lsent) == len(lpos)
        assert len(rsent) == len(rpos)

        sent = lsent + rsent + psent
        pos = lpos + rpos + ppos

        assert len(sent) == args.ngbrs * 4 + 2 
        assert len(pos) == args.ngbrs * 4 + 2
        
        return (sent, pos, ex.rev, lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, pred_ind)

    def create_features_tbd(self, ex, tokenizer, args):
        ### create a new feature creation method for TBD dataset                                                               

        # compute token index                                                                                                  
        pos2idx = args.pos2idx
        pos_dict = ex.doc.pos_dict                                                                                            \

        all_keys, lidx_start, lidx_end, ridx_start, ridx_end = token_idx(ex.left.span, ex.right.span, pos_dict)

        # find the start and end point of a sentence                                                                           
        left_seq = [pos_dict[x][0] for x in all_keys[:lidx_start]]
        right_seq = [pos_dict[x][0] for x in all_keys[ridx_end + 1:]]

        try:
            sent_start = max(loc for loc, val in enumerate(left_seq) if val == '.') + 1
        except:
            sent_start = 0

        # sent_end token will not be included                                                                                  
        try:
            sent_end = ridx_end + 1 + min(loc for loc, val in enumerate(right_seq) if val == '.')
        except:
            sent_end = len(pos_dict)

        assert sent_start < sent_end
        assert sent_start <= lidx_start
        assert ridx_end <= sent_end

        sent_key = all_keys[sent_start:sent_end]

        # construct sentence with bert tokenizer index                                                                       
        pos = [pos2idx[k] if k in pos2idx.keys() else len(pos2idx) for k in [pos_dict[x][1] for x in sent_key]]

        #print(sent_key)
        tokenizer = tokenizer

        # NOTE: this is a quick way to hack bert tokenization
        # need to figure out a way in data ingestion process
        sent = []
        for x in [pos_dict[x][0].lower() for x in sent_key]:
            try:
                sent += tokenizer.convert_tokens_to_ids([x])
            except:
                sent += tokenizer.convert_tokens_to_ids(['[UNK]'])

        #print(sent)
        #kill
        # create lexical features for the model                                                                                
        new_fts = []
        new_fts.append(-distance_features(lidx_start, lidx_end, ridx_start, ridx_end))
        new_fts.extend(polarity_features(ex.left, ex.right))
        new_fts.extend(tense_features(ex.left, ex.right))

        # prediction is always true for TCR dataset                                                                            
        pred_ind = True

        # calculate events' index in sentence                                                                                                                      
        lidx_start_s = lidx_start - sent_start
        lidx_end_s = lidx_end - sent_start
        ridx_start_s = ridx_start - sent_start
        ridx_end_s = ridx_end - sent_start

        return (sent, pos, new_fts, ex.rev, lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, pred_ind)


    def create_features_tcr(self, ex, args):
        ### create a new feature creation method for TCR dataset
        
        # compute token index
        pos2idx = args.pos2idx
        pos_dict = ex.doc.pos_dict                                                                                                                                                   
        all_keys, lidx_start, lidx_end, ridx_start, ridx_end = token_idx(ex.left.span, ex.right.span, pos_dict)

        # find the start and end point of a sentence
        left_seq = [pos_dict[x][0] for x in all_keys[:lidx_start]]
        right_seq = [pos_dict[x][0] for x in all_keys[ridx_end + 1:]]

        try:
            sent_start = max(loc for loc, val in enumerate(left_seq) if val == '.') + 1
        except:
            sent_start = 0
        
        # sent_end token will not be included
        try:
            sent_end = ridx_end + 1 + min(loc for loc, val in enumerate(right_seq) if val == '.')
        except:
            sent_end = len(pos_dict)

        assert sent_start < sent_end
        assert sent_start <= lidx_start
        assert ridx_end <= sent_end

        sent_key = all_keys[sent_start:sent_end]
        # construct sentence and pos sequence with index
        sent = [args.w2i[t] if t in args.w2i.keys() else 1 for t in [pos_dict[x][0].lower() for x in sent_key]]
        pos = [pos2idx[k] if k in pos2idx.keys() else len(pos2idx) for k in [pos_dict[x][1] for x in sent_key]]

        # create lexical features for the model
        new_fts = []
        new_fts.append(-distance_features(lidx_start, lidx_end, ridx_start, ridx_end))
        new_fts.extend(polarity_features(ex.left, ex.right))
        new_fts.extend(tense_features(ex.left, ex.right))
        
        # prediction is always true for TCR dataset 
        pred_ind = True

        # calculate events' index in sentence
        lidx_start_s = lidx_start - sent_start
        lidx_end_s = lidx_end - sent_start
        ridx_start_s = ridx_start - sent_start
        ridx_end_s = ridx_end - sent_start
        
        return (sent, pos, new_fts, ex.rev, lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, pred_ind)


    def parallel(self, ex, tokenizer, args):

        if args.data_type == "red":
            return ex.doc_id, ex.id, self._label_to_id[ex.rel_type], self.create_features(ex, args)
        elif args.data_type == "new":
            if ex.id[0] == 'C':
                return ex.doc_id, ex.id, self._label_to_id_c[ex.rel_type], self.create_features_tcr(ex, args)
            else:
                return ex.doc_id, ex.id, self._label_to_id[ex.rel_type], self.create_features_tcr(ex, args)
        # matres dataset is based on tbd data
        elif args.data_type in ["matres", "tbd"]:
            return ex.doc_id, ex.id, self._label_to_id[ex.rel_type], self.create_features_tbd(ex, tokenizer, args)

    def predict(self, model, data, args):
        model.eval()
        criterion = nn.CrossEntropyLoss()                                                                                                         
        eval_batch = 1

        sents, poss, labels, ftss, pred_inds = [], [], [], [], []

        count = 1
        correct = 0
        
        for data_id, label, (sent, pos, new_fts, rev, lidx_start, lidx_end, ridx_start, ridx_end, pred_ind) in data:

            sents.append(sent)
            poss.append(pos)
            labels.append(label)
            ftss.append(new_fts)
            pred_inds.append(pred_ind)

            if count % eval_batch == 0 or count == len(data):
                sents = (Variable(torch.LongTensor(np.array(sents).transpose())), 
                         Variable(torch.LongTensor(np.array(poss).transpose())),
                         Variable(torch.FloatTensor(np.array(ftss))))

                labels = Variable(torch.LongTensor(np.array(labels)))
                
                is_causal = (data_id[0] == 'C')
                out, prob = model(labels, sents, lidx_start, lidx_end, ridx_start, ridx_end, pred_inds,causal=is_causal)
                out = out.view(labels.size()[0], -1)
                
                loss = criterion(out, labels)

                # first sample: initialize lists
                if count == eval_batch:
                    # the first sample in evalset shouldn't be causal (no shuffle)
                    assert data_id[0] != 'C'
                    losses_t = [loss.data]
                    losses_c = []
                    probs = [prob]
                # temporal case
                elif not is_causal:
                    losses_t.append(loss.data)
                    probs.append(prob)
                # causal case
                else:
                    predicted = (prob.data.max(1)[1]).long().view(-1)
                    correct += (predicted == labels.data).sum() 
                    losses_c.append(loss.data)

                sents, poss, labels, ftss, pred_inds = [], [], [], [], []
                
            count += 1
            if count % 1000 == 0:
                print("finished evaluating %s samples" % count)
        probs = torch.cat(probs,dim=0)
        print("Evaluation temporal loss: %.4f" % np.mean(losses_t))
        if args.joint:
            print("Evaluation causal loss: %.4f; accuracy: %.4f" % (np.mean(losses_c), float(correct) / float(len(losses_c))))
        return probs.data, np.mean(losses_t)
        #return (probs.data.max(1)[1]).long().view(-1), np.mean(losses)


    def _train(self, train_data, eval_data, pos_emb, args):
        # since we are training with single sample in each iteration,
        # we don't need to distinguis causal and temporal labels
         
        if args.model == 'bert':
            config = BertConfig(**args.bert_config)
            model = BertClassifier(config, args)

        # fix bert model parameters
        if not args.fine_tune:
            for name, param in model.named_parameters():
                if 'bert' in name:
                    param.requires_grad = False

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        criterion = nn.CrossEntropyLoss()                                                                                                                                                
        losses = [] 

        sents, poss, ftss, labels = [], [], [], []                                                                                                                                                 
        best_eval_f1 = 0.0 
        
        for epoch in range(args.epochs):
            print("Training Epoch #%s..." % (epoch + 1))
            model.train()
            correct = 0
            count = 1
            loss_hist_t, loss_hist_c = [], []

            start_time = time.time()
            for data_id, label, (sent, pos, new_fts, rev, lidx_start, lidx_end, ridx_start, ridx_end, _) in train_data:
                
                sents.append(sent)                                                                                                                                                       
                poss.append(pos)        
                ftss.append(new_fts)
                labels.append(label) 
                
                # batch computation                                                                                                                                                            
                if count % args.batch == 0 or count == total_samples:                                                                                                                          
                    sents = (Variable(torch.LongTensor(np.array(sents).transpose())), 
                             Variable(torch.LongTensor(np.array(poss).transpose())),
                             Variable(torch.FloatTensor(np.array(ftss))))
                                                  
                    labels = Variable(torch.LongTensor(np.array(labels)))                                                                                                                                                                                               
                    model.zero_grad()                                                                                                                                 
                    is_causal = (data_id[0] == 'C')
                    out, prob = model(labels, sents, lidx_start, lidx_end, ridx_start, ridx_end, flip=rev, causal=is_causal)                                                                                                    
                    out = out.view(labels.size()[0], -1)                                                                                     
                    loss = criterion(out, labels)                                                                                                                                              
                    loss.backward()                                                                                                                                                            
                    #torch.nn.utils.clip_grad_norm(model.parameters(), args.clipper)                                                                                                       
                    optimizer.step()                                                                                                                                                      
                    predicted = (prob.data.max(1)[1]).long().view(-1)
                    
                    if is_causal:
                        correct += (predicted == labels.data).sum()                                                                                                                                                                 
                    sents, poss, ftss, labels = [], [], [], []                                                                                                                                                                              
                if args.batch == 1:                                                                                                                                         
                    if is_causal:
                        loss_hist_c.append(loss.data.numpy())
                    else:    
                        loss_hist_t.append(loss.data.numpy())                                                                                                                                        
                else:                                                                                                                                       
                    loss_hist.extend(loss.data.numpy().tolist())                                                                                                                                                                                               
                if count % 1000 == 0:                                                                                                       
                    print("trained %s samples" % count)
                    print("Temporal loss is %.4f" % np.mean(loss_hist_t))
                    if args.joint:
                        print("Causal loss is %.4f; accuracy is %.4f" % (np.mean(loss_hist_c), float(correct) / float(len(loss_hist_c)))) 
                    print("%.4f seconds elapsed" % (time.time() - start_time))                                                              
                count += 1 
            # Evaluate at the end of each epoch                                 
                                                                                 
            print("*"*50)
            print(len(eval_data))
            # select model only based on temporal F1 if refit on train / eval split
            # if doing final fitting based on train and epoch from CV, then only use all train data
            if len(eval_data) > 0:
                eval_labels = torch.LongTensor(np.array([x[1] for x in eval_data if x[0][0] != 'C']))
                eval_preds, eval_loss = self.predict(model, eval_data, args)
                pred_labels = eval_preds.max(1)[1].long().view(-1)
            
                assert eval_labels.size() == pred_labels.size() 
                eval_correct = (pred_labels == eval_labels).sum()
                eval_acc =  float(eval_correct) / float(len(eval_labels))
                eval_f1 = self.weighted_f1(list(pred_labels.numpy()), list(eval_labels.numpy()))

                if eval_f1 >= best_eval_f1:
                    best_eval_f1 = eval_f1
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch

                print("Evaluation loss: %.4f; Evaluation F1: %.4f" % (eval_loss, eval_f1))
                print("*"*50)

        print("Final Evaluation F1: %.4f" % best_eval_f1)
        print("*"*50)

        if len(eval_data) > 0:
            self.model = best_model
        else:
            self.model = copy.deepcopy(model)
            best_epoch = epoch

        if args.save_model == True:
            torch.save({'epoch': epoch,
                        'args': args,
                        'state_dict': self.model.state_dict(),
                        'f1': best_eval_f1,
                        'optimizer' : optimizer.state_dict()
                    }, "%s%s.pth.tar" % (args.ilp_dir, args.save_stamp))
        
        return best_eval_f1, best_epoch
    
    def data_split(self, train_docs, eval_docs, data, nr):

        train_set = []
        eval_set = []
        
        for s in data:
            if s[0] in eval_docs:
                eval_set.append((s[1], s[2], s[3]))
            elif s[1][0] in ['P', 'C']:
                train_set.append((s[1], s[2], s[3]))
            # for training set, only keep nr negative samples
            # high = total negative / total positive
            # low = 0
            elif nr > np.random.uniform(high=10):
                train_set.append((s[2], s[3]))
        
        return train_set, eval_set

    def cross_validation(self, data, emb, pos_emb, args):
        
        kf = KFold(n_splits=args.n_splits, random_state=42)

        param_perf = []
        for param in ParameterGrid(args.params):
            param_str = ""
            for k,v in param.items():
                exec("args.%s=%s" % (k, v))
                param_str += "%s=%s" % (k, v)
                param_str += " "

            print("Train parameters: %s" % param_str)
            print("*" * 50)

            for k,v in param.items():
                exec("args.%s=%s" % (k, v))
            
            trn_docs = args.train_docs
            dev_docs = args.dev_docs
            all_docs = trn_docs + dev_docs

            all_splits = [{'train':trn, 'eval':evl} for trn,evl in kf.split(all_docs)]
            assert len(all_splits) == args.n_splits

            # multi-process over only data splits due to source limitation
            with mp.Pool(processes=args.n_splits) as pool:
                res = pool.map(partial(self.parallel_cv, all_docs=all_docs, data=data, emb=emb, pos_emb=pos_emb, args=args), all_splits)
            print(res)
            f1s = list(zip(*res))[0]
            best_epoch = list(zip(*res))[1]
            param_perf.append((param, np.mean(f1s), np.mean(best_epoch)))

        params, f1, epoch = sorted(param_perf, key=lambda x: x[1], reverse=True)[0]
        print(sorted(param_perf, key=lambda x: x[1], reverse=True))
        print("Best Average F1: %s" % f1)
        print("Best Parameters Are: %s " % params)
        print("Best Epoch is: %s" % epoch)
        return params

    def parallel_cv(self, train_eval_docs, all_docs=[], data = [], emb = np.array([]), pos_emb = [], args=None):
        
        train_doc = train_eval_docs['train']
        eval_doc = train_eval_docs['eval']
        
        print("# of train docs: %s; # of eval docs: %s" % (len(train_doc), len(eval_doc)))
        train_data, eval_data = self.data_split([all_docs[x] for x in train_doc], [all_docs[x] for x in eval_doc], data, args.nr)
        print("# of train samples: %s; # of eval samples: %s" % (len(train_data), len(eval_data)))
        return self._train(train_data, eval_data, emb, pos_emb, args)
                  
    def train_epoch(self, 
                    train_data: Iterator[FlatRelation], 
                    args) -> float:

        if args.data_type == "red":
            label_map = red_label_map
        elif args.data_type == "matres":
            label_map = matres_label_map
        elif args.data_type == "tbd":
            label_map = tbd_label_map
        else:
            label_map = new_label_map

        all_labels = list(OrderedDict.fromkeys(label_map.values()))

        if args.joint:
            label_map_c = causal_label_map
            # in order to perserve order of unique keys
            all_labels_c =  list(OrderedDict.fromkeys(label_map_c.values()))
            self._label_to_id_c = OrderedDict([(all_labels_c[l],l) for l in range(len(all_labels_c))])
            self._id_to_label_c = OrderedDict([(l,all_labels_c[l]) for l in range(len(all_labels_c))])
            print(self._label_to_id_c)
            print(self._label_to_id_c)

        self._label_to_id = OrderedDict([(all_labels[l],l) for l in range(len(all_labels))])
        self._id_to_label = OrderedDict([(l,all_labels[l]) for l in range(len(all_labels))])

        print(self._label_to_id)
        print(self._id_to_label)

        args.label_to_id = self._label_to_id

        #emb = np.array(list(args.glove.values()))
        #emb = np.vstack((np.random.uniform(0, 1, (2, emb.shape[1])), emb))

        #assert emb.shape[0] == vocab.shape[0]

        pos_emb= np.zeros((len(args.pos2idx) + 1, len(args.pos2idx) + 1))
        for i in range(pos_emb.shape[0]):
            pos_emb[i, i] = 1.0

        # for debug
        #for ex in train_data:
        #    print(self.create_features_tbd(ex, args))
        #    kill
        ### collect data features before running models
        with mp.Pool(processes=2) as pool:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            data = pool.map(partial(self.parallel, tokenizer=tokenizer, args=args), train_data)
        print(len(data))


        if args.cv == True:
            best_params = self.cross_validation(data, emb, pos_emb, args)

            ### retrain on the best parameters
            print("To refit...")
            args.refit = True

            for k,v in best_params.items():
                exec("args.%s=%s" % (k, v))
        
        if args.data_type in ['new']:
            train_docs, dev_docs = train_test_split(args.train_docs, test_size=0.2, random_state=7)

        # Both RED and TBDense data have give splits on train/dev/test
        else:
            train_docs = args.train_docs 
            dev_docs = args.dev_docs

        print(len(train_docs), len(dev_docs))
        if args.refit_all:
            train_docs = args.train_docs
            dev_docs = []
        
        train_data, dev_data = self.data_split(train_docs, dev_docs, data, args.nr)
        print(len(train_data), len(dev_data))

        best_f1, _ = self._train(train_data, dev_data, pos_emb, args)
        print("Final Dev F1: %.4f" % best_f1)
        return -1.0

    def weighted_f1(self, pred_labels, true_labels):
        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else numr / denr

        assert len(pred_labels) == len(true_labels)

        weighted_f1_scores = {}
        if 'NONE' in self._label_to_id.keys():
            num_tests = len([x for x in true_labels if x != self._label_to_id['NONE']])
        else:
            num_tests = len([x for x in true_labels])

        print("Total samples to eval: %s" % num_tests)
        total_true = Counter(true_labels)
        total_pred = Counter(pred_labels)

        labels = list(self._id_to_label.keys())
        
        n_correct = 0
        n_true = 0
        n_pred = 0

        for label in labels:
            if self._id_to_label[label] not in ['NONE', 'VAGUE']:
                
                true_count = total_true.get(label, 0)
                pred_count = total_pred.get(label, 0)
        
                n_true += true_count
                n_pred += pred_count
                
                correct_count = len([l for l in range(len(pred_labels)) 
                                     if pred_labels[l] == true_labels[l] and pred_labels[l] == label])
                #for l in range(len(pred_labels)):
                #    if pred_labels[l] == true_labels[l] and pred_labels[l] == label:
                #        correct_count += 1
                n_correct += correct_count
                #precision = safe_division(correct_count, pred_count)
                #recall = safe_division(correct_count, true_count)

        precision = safe_division(n_correct, n_pred)                                                               
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2 * precision * recall, precision + recall)
                #weight = safe_division(true_count, num_tests)
                #weighted_f1_scores[self._id_to_label[label]] = f1_score * weight
        return(f1_score)
        
        #print(weighted_f1_scores)
        #return sum(list(weighted_f1_scores.values()))

@dataclass
class REDEvaluator:
    model: REDEveEveRelModel
    def evaluate(self, test_data: Iterator[FlatRelation], args):
        # load test data first since it needs to be executed twice in this function                                                

        to_ilp = {}
        pairs = [(ex.doc.id+"_"+ex.left.id, ex.doc.id+"_"+ex.right.id) for ex in test_data]
        to_ilp['pairs'] = pairs

        with mp.Pool(processes=2) as pool:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            data = pool.map(partial(self.model.parallel, tokenizer=tokenizer, args=args), test_data)

        true_labels = [x.rel_type for x in test_data]
        to_ilp['labels'] = true_labels

        preds, loss = self.model.predict(self.model.model, [(x[1], x[2], x[3]) for x in data], args)
        
        to_ilp['probs'] = preds.numpy()

        out_path = Path(args.ilp_dir+'ilp.pkl')
        with out_path.open('wb') as fh:
            pickle.dump(to_ilp, fh)

        preds = (preds.max(1)[1]).long().view(-1)
        pred_labels = [self.model._id_to_label[x] for x in preds.numpy().tolist()]

        true_labels = [x.rel_type for x in test_data if x.id[0] != 'C']
        return ClassificationReport(self.model.name, true_labels, pred_labels)

def main(args):

    data_dir = args.data_dir
    opt_args = {}

    if args.tempo_filter:
        opt_args['include_types'] = args.include_types 
    if args.skip_other:
        opt_args['other_label'] = None
    if args.nr > 0:
        opt_args['neg_rate'] = args.nr
        opt_args['eval_list'] = args.eval_list

    opt_args['data_type'] = args.data_type
    opt_args['pred_window'] = args.pred_win
    opt_args['shuffle_all'] = args.shuffle_all
    opt_args['joint'] = args.joint
    log.info(f"Reading datasets to memory from {data_dir}")
    # buffering data in memory --> it could cause OOM
    opt_args['backward_sample'] = False
    train_data = list(read_relations(Path(data_dir), 'train', **opt_args))
    dev_data = []
    
    if args.data_type in ["red", "tbd", "matres"]:
        dev_data = list(read_relations(Path(data_dir), 'dev', **opt_args))
    
    # combine train and dev
    # either CV or split by specific file name later
    train_data += dev_data

    test_data = list(read_relations(Path(data_dir), 'test', **opt_args))
    models = [NNClassifier()]
    for model in models:
        print(f"\n======={model.name}=====")
        print(f"======={args.model}=====\n")
        model.train_epoch(train_data, args)
        evaluator = REDEvaluator(model)
        print(evaluator.evaluate(dev_data, args))
        print(evaluator.evaluate(test_data, args))
    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # arguments for data processing
    p.add_argument('-data_dir', type=str,
                   help='Path to directory having RED data. This should be output of '
                        '"ldcred.py flexnlp"')
    p.add_argument('--tempo_filter', action='store_true',
                   help='Include Causal and Temporal relations only. By default all relations are'
                        ' included. When --filter is specified, non temporal and causal relations '
                        'will be labelled as OTHER')
    p.add_argument('--skip_other', action='store_true',
                   help='when --tempo-filter is applied, the excluded types are marked as OTHER.'
                        'By enabling --skip-other, OTHER types are skipped.')
    
    p.add_argument('-nr', '--neg-rate', type=float, default=0.,
                   help='Negative sample rate.')

    p.add_argument('-include_types', type=set, default={'TLINK'})
    p.add_argument('-eval_list', type=list, default=[])
    p.add_argument('-shuffle_all', type=bool, default=False)
    # select model
    p.add_argument('-model', type=str, default='bert')

    # arguments for RNN model
    p.add_argument('-emb', type=int, default=100)
    p.add_argument('-hid', type=int, default=20)
    p.add_argument('-num_layers', type=int, default=1)
    p.add_argument('-batch', type=int, default=1)
    p.add_argument('-data_type', type=str, default="red")
    p.add_argument('-epochs', type=int, default=10)
    p.add_argument('-seed', type=int, default=123)
    p.add_argument('-lr', type=float, default=0.0005)
    p.add_argument('-num_classes', type=int, default=2) # get updated in main()
    p.add_argument('-dropout', type=float, default=0.5)
    p.add_argument('-ngbrs', type=int, default = 15)                                   
    p.add_argument('-pos2idx', type=dict, default = {})
    p.add_argument('-w2i', type=OrderedDict)
    p.add_argument('-glove', type=OrderedDict)
    p.add_argument('-cuda', type=bool, default=False)
    p.add_argument('-refit_all', type=bool, default=False)
    p.add_argument('-params', type=dict, default={})
    p.add_argument('-n_splits', type=int, default=5)
    p.add_argument('-pred_win', type=int, default=200)
    p.add_argument('-n_fts', type=int, default=15)
    # arguments for CNN model
    p.add_argument('-stride', type=int, default = 1)
    p.add_argument('-kernel', type=int, default = 5)
    p.add_argument('-train_docs', type=list, default=[])
    p.add_argument('-dev_docs', type=list, default=[])
    p.add_argument('-cv', type=bool, default=False)
    p.add_argument('-attention', type=bool, default=False)
    p.add_argument('-save_model', type=bool, default=False)
    p.add_argument('-save_stamp', type=str, default="")
    p.add_argument('-ilp_dir', type=str, default="/nas/home/rujunhan/ILP/")
    p.add_argument('-joint', type=bool, default=False)
    p.add_argument('-num_causal', type=int, default=2)
    p.add_argument('-bert_config', type=dict, default={})
    p.add_argument('-fine_tune', type=bool, default=False)
    args = p.parse_args()
    
    #args.eval_list = ['train', 'dev', 'test']
    
    args.eval_list = []
    args.data_type = "tbd"
    if args.data_type == "red":
        args.data_dir = "/nas/home/rujunhan/red_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args.data_dir, 'r')]
    elif args.data_type == "new":
        args.data_dir = "/nas/home/rujunhan/tcr_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')] 
    elif args.data_type == "matres":
        args.data_dir = "/nas/home/rujunhan/matres_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args.data_dir, 'r')]
    elif args.data_type == "tbd":
        args.data_dir = "/nas/home/rujunhan/tbd_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args.data_dir, 'r')]

    # create pos_tag and vocabulary dictionaries
    # make sure raw data files are stored in the same directory as train/dev/test data
    
    tags = open("/nas/home/rujunhan/tcr_output/pos_tags.txt")
    pos2idx = {}
    idx = 0
    for tag in tags:
        tag = tag.strip()
        pos2idx[tag] = idx
        idx += 1
    args.pos2idx = pos2idx
    args.cuda = False
    
    args.pred_win = args.ngbrs * 2

    args.bert_config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "type_vocab_size": 2,
        "vocab_size_or_config_json_file": 30522
    }
    #args.glove = read_glove("/nas/home/rujunhan/red_output/glove/glove.6B.%sd.txt" % args.emb)
    #vocab = np.array(['<pad>', '<unk>'] + list(args.glove.keys()))
    #args.w2i = OrderedDict((vocab[i], i) for i in range(len(vocab)))

    args.nr = 0.0
    args.tempo_filter = True
    args.skip_other = True
    
    args.params = {'hid': [50], 'dropout': [0.5, 0.6]}
    
    main(args)
