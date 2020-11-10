from pathlib import Path
import pickle
import sys
import argparse
from collections import defaultdict, Counter, OrderedDict
from typing import Iterator, List, Mapping, Union, Optional, Set
from dataclasses import dataclass
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import copy
from torch.utils import data
from base import EveEveRelModel, matres_label_map, tbd_label_map, new_label_map, causal_label_map
from base import  ClassificationReport, rev_map, rev_causal_map
from featureFuncs import *
from gurobi_inference import Gurobi_Inference
import multiprocessing as mp
from functools import partial
from sklearn.model_selection import ParameterGrid
from temporal_evaluation import *
from nn_model import BiLSTM
from dataloader import get_data_loader
from dataset import EventDataset
import os
 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@dataclass
class Evaluator:
    model: EveEveRelModel
    def evaluate(self, test_data, args):
        true_labels, pred_labels = self.model.predict(self.model.model, test_data, args, in_dev=False)
        pred_labels = [self.model._id_to_label[x] for x in pred_labels]
        true_labels = [self.model._id_to_label[x] for x in true_labels]

        assert len(pred_labels) == len(true_labels)
        print(len(pred_labels))
        if args.data_type in ['tbd']:
            organized_test_data = []
            for x in test_data:
                for i in range(x[0].size(0)):
                    if args.teston=='bothway':
                        organized_test_data.append((x[2][0][i], x[2][1][i][0], x[2][1][i][1], self.model._id_to_label[(x[3].cpu().tolist())[i]]))
                    elif args.teston=='forward':
                        if x[7][i]==False:
                            organized_test_data.append((x[2][0][i], x[2][1][i][0], x[2][1][i][1], self.model._id_to_label[(x[3].cpu().tolist())[i]]))
                    elif args.teston=='backward':
                        if x[7][i]==True:
                            organized_test_data.append((x[2][0][i], x[2][1][i][0], x[2][1][i][1], self.model._id_to_label[(x[3].cpu().tolist())[i]]))

            temporal_awareness(organized_test_data, pred_labels, args.data_type, args.eval_with_timex)
        
        if args.data_type == 'tbd':
            return ClassificationReport(self.model.name, true_labels, pred_labels, False)
        else:
            return ClassificationReport(self.model.name, true_labels, pred_labels)
    
    def get_score(self, test_data, args):
        true_labels, pred_labels = self.model.predict(self.model.model, test_data, args, in_dev=False)
        return self.model.weighted_f1(pred_labels, true_labels)

    def collect_result(self, test_data, args):
        # collect results that used for McNemar Test
        true_labels, pred_labels = self.model.predict(self.model.model, test_data, args, in_dev=False)
        pred_labels = [self.model._id_to_label[x] for x in pred_labels]
        true_labels = [self.model._id_to_label[x] for x in true_labels]
        
        matrix = {}
        idx = 0
        for x in test_data:
            for i in range(x[0].size(0)):
                if args.teston=='forward':
                    if (x[7][i]==False) and (x[1][i][0]=='L'):
                        correctness = (pred_labels[idx]==true_labels[idx])
                        name = 'global_seed'+str(100)+'_'+str(x[2][0][i])+'_'+str(x[2][1][i][0])+'_'+str(x[2][1][i][1])
                        matrix[name]=correctness
                        idx+=1
        assert(idx==len(pred_labels))
        filename = 'global_'+args.data_type+'_seed'+str(100)+'.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(matrix, f)
        return

@dataclass()
class NNClassifier(EveEveRelModel):
    _label_to_id_c = {}
    _id_to_label_c = {}

    def predict(self, model, eval_data, args, in_dev=False):
        model.eval()
        step = 1
        correct = 0.
        eval_pairs = []
        eval_pairs_r = []
        eval_pairs_c = []
        eval_pairs_c_r = []
        probs, probs_r, probs_c, probs_c_r = [], [], [], []
        gt_labels, gt_labels_r, gt_labels_c, gt_labels_c_r = [], [], [], []
        for data in eval_data:
            seq_lens,data_id,(doc_ids,pairs),labels,sents,poss,fts,revs,lidx_start,lidx_end,ridx_start,ridx_end,_ = togpu_data(data)
            idx_c = []
            idx_c_r = []
            idx_l = []
            idx_l_r = []
            for i, ids in enumerate(data_id):
                if ids[0] == 'C':
                    if revs[i]:
                        idx_c_r.append(i)
                    else:
                        idx_c.append(i)
                elif ids[0] == 'L':
                    if revs[i]:
                        idx_l_r.append(i)
                    else:
                        idx_l.append(i)
            if len(idx_l) > 0:
                seq_l = seq_lens[idx_l]
                sent = sents[idx_l]
                pos = poss[idx_l]
                ft = fts[idx_l]
                l_start = lidx_start[idx_l]
                l_end = lidx_end[idx_l]
                r_start = ridx_start[idx_l]
                r_end = ridx_end[idx_l]
                out, prob = model(seq_l, (sent, pos, ft), l_start, l_end, 
                                  r_start, r_end, flip=False, causal=False)
                label = labels[idx_l]
                doc_id = [doc_ids[i] for i in idx_l]
                pair = [pairs[i] for i in idx_l]
                for i in range(len(doc_id)):
                    left = pair[i][0]
                    right = pair[i][1]
                    eval_pairs.append(("%s_%s"%(doc_id[i], left), "%s_%s"%(doc_id[i], right)))
                probs.append(prob)
                gt_labels.append(label)

            if len(idx_l_r) > 0:
                seq_l = seq_lens[idx_l_r]
                sent = sents[idx_l_r]
                pos = poss[idx_l_r]
                ft = fts[idx_l_r]
                l_start = lidx_start[idx_l_r]
                l_end = lidx_end[idx_l_r]
                r_start = ridx_start[idx_l_r]
                r_end = ridx_end[idx_l_r]
                out, prob = model(seq_l, (sent, pos, ft), l_start, l_end, 
                                  r_start, r_end, flip=True, causal=False)
                label = labels[idx_l_r]
                doc_id = [doc_ids[i] for i in idx_l_r]
                pair = [pairs[i] for i in idx_l_r]
                for i in range(len(doc_id)):
                    left = pair[i][0]
                    right = pair[i][1]
                    eval_pairs_r.append(("%s_%s"%(doc_id[i], right), "%s_%s"%(doc_id[i], left)))
                probs_r.append(prob)
                gt_labels_r.append(label)
                
            if (len(idx_c) > 0) and args.joint:
                seq_l = seq_lens[idx_c]
                sent = sents[idx_c]
                pos = poss[idx_c]
                ft = fts[idx_c]
                l_start = lidx_start[idx_c]
                l_end = lidx_end[idx_c]
                r_start = ridx_start[idx_c]
                r_end = ridx_end[idx_c]
                out, prob = model(seq_l, (sent, pos, ft), l_start, l_end, 
                                  r_start, r_end, flip=False, causal=True)
                label = labels[idx_c]
                predicted = (prob.data.max(1)[1]).long().view(-1)
                correct += (predicted == label.data).sum()
                doc_id = [doc_ids[i] for i in idx_c]
                pair = [pairs[i] for i in idx_c]
                for i in range(len(doc_id)):
                    left = pair[i][0]
                    right = pair[i][1]
                    eval_pairs_c.append(("%s_%s"%(doc_id[i], left), "%s_%s"%(doc_id[i], right)))
                probs_c.append(prob)
                gt_labels_c.append(label)
            
            if (len(idx_c_r) > 0) and args.joint:
                seq_l = seq_lens[idx_c_r]
                sent = sents[idx_c_r]
                pos = poss[idx_c_r]
                ft = fts[idx_c_r]
                l_start = lidx_start[idx_c_r]
                l_end = lidx_end[idx_c_r]
                r_start = ridx_start[idx_c_r]
                r_end = ridx_end[idx_c_r]
                out, prob = model(seq_l, (sent, pos, ft), l_start, l_end, 
                                  r_start, r_end, flip=True, causal=True)
                label = labels[idx_c_r]
                predicted = (prob.data.max(1)[1]).long().view(-1)
                correct += (predicted == label.data).sum()
                doc_id = [doc_ids[i] for i in idx_c_r]
                pair = [pairs[i] for i in idx_c_r]
                for i in range(len(doc_id)):
                    left = pair[i][0]
                    right = pair[i][1]
                    eval_pairs_c_r.append(("%s_%s"%(doc_id[i], right), "%s_%s"%(doc_id[i], left)))
                probs_c_r.append(prob)
                gt_labels_c_r.append(label)
        # perform global inference
        # concat all data first
        eval_pairs = eval_pairs+eval_pairs_r
        eval_pairs_c = eval_pairs_c+eval_pairs_c_r
        if args.trainon=='bothWselect':
            # perform selection on local model
            max_probs_f = torch.cat(probs, dim=0).max(1)[0].reshape(-1,1)
            max_probs_b = torch.cat(probs_r, dim=0).max(1)[0].reshape(-1,1)
            max_probs = torch.cat((max_probs_f, max_probs_b), 1)
            mask = max_probs.max(1)[1].view(-1)
            probs_forward = []
            probs_backward = []
            for i, j in enumerate(mask):
                if j == 0:
                    probs_forward.append(probs[i])
                    probs_backward.append(self.get_reverse_prob(probs[i]))
                else:
                    probs_forward.append(self.get_reverse_prob(probs_r[i]))
                    probs_backward.append(probs_r[i])
            probs = torch.stack((probs_forward, probs_backward), dim=0)
        else:
            probs = torch.cat((probs+probs_r), dim=0)
        prob_table = probs.cpu().data.numpy()
        gt_labels = torch.cat(gt_labels+gt_labels_r, dim=0)
        prob_table_c = np.zeros((0, 0))
        ground_truth = gt_labels
        if (len(probs_c)>0) and (args.joint):
            if args.trainon=='bothWselect':
                max_probs_f_c = torch.cat(probs_c, dim=0).max(1)[0].reshape(-1,1)
                max_probs_b_c = torch.cat(probs_c_r, dim=0).max(1)[0].reshape(-1,1)
                max_probs_c = torch.cat((max_probs_f_c,max_probs_b_c), dim=1)
                mask = max_probs_c.max(1)[1].view(-1)
                probs_c_forward = []
                probs_c_backward = []
                for i, j in enumerate(mask):
                    if j == 0:
                        probs_c_forward.append(probs_c[i])
                        probs_c_backward.append(self.get_reverse_prob_c(probs_c[i]))
                    else:
                        probs_c_forward.append(self.get_reverse_prob_c(probs_c_r[i]))
                        probs_c_backward.append(probs_c_r[i])
                probs_c = torch.stack((probs_c_forward, probs_c_backward), dim=0)
            else:
                probs_c = torch.cat((probs_c+probs_c_r), dim=0)
            prob_table_c = probs_c.cpu().data.numpy()
            gt_labels_c = torch.cat(gt_labels_c+gt_labels_c_r, dim=0)
            ground_truth = torch.cat((gt_labels, gt_labels_c), dim=0)
        # find max prediction based on global prediction 
        best_pred_idx, best_pred_idx_c, predictions=\
            self.global_prediction(eval_pairs, prob_table, eval_pairs_c,
                                   prob_table_c, evaluate=True,
                                   true_labels=ground_truth, backward=(args.trainon!='forward'),
                                   trans_only=(args.trans_only))
        loss = self.loss_func(best_pred_idx, gt_labels, probs, args.margin)
        print("Evaluation loss: %.4f" % loss.cpu().data.numpy())
        print("*"*50)
        N = len(eval_pairs)
        if in_dev and args.devbytrain:
            if (args.trainon=='forward'):
                final_pred_labels = predictions[:int(N/2)]
                final_gt_labels = gt_labels[:int(N/2)]
            else:
                final_pred_labels = predictions[:N]
                final_gt_labels = gt_labels[:N]

        else:
            if args.teston=='forward':
                final_pred_labels = predictions[:int(N/2)]
                final_gt_labels = gt_labels[:int(N/2)]
            elif args.teston=='backward':
                final_pred_labels = predictions[int(N/2):N]
                final_gt_labels = gt_labels[int(N/2):N]
            elif args.teston=='bothway':
                final_pred_labels = predictions[:N]
                final_gt_labels = gt_labels[:N]

        return final_gt_labels.cpu().tolist(), final_pred_labels.cpu().tolist()

    def _train(self, train_data, eval_data, emb, pos_emb, args, in_cv=False):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        model = BiLSTM(emb, pos_emb, args)
        if args.cuda and torch.cuda.is_available():
            model = togpu(model)
        if args.load_model == True:
            checkpoint = torch.load(args.ilp_dir + args.load_model_file)
            model.load_state_dict(checkpoint['state_dict'])
            best_eval_f1 = checkpoint['f1']
            print("Local best eval f1 is: %s" % best_eval_f1)
        best_epoch = 0
        best_eval_f1 = 0.0
        
        if args.optimizer == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
        elif args.optimizer == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=args.lr, weight_decay=args.decay)
        else:
            raise ValueError('the choice of optimizer did not exist')
        early_stop_counter = 0 
        for epoch in range(args.epochs):
            if not in_cv:
                print('Training... %s-th epoch'%(epoch+1)) 
            model.train()
            correct = 0.
            step = 1    
            train_pairs = []
            train_pairs_r = []
            train_pairs_c = []
            train_pairs_c_r = []
            probs, probs_r, probs_c, probs_c_r = [], [], [], []
            gt_labels, gt_labels_r, gt_labels_c, gt_labels_c_r = [], [], [], []
            start_time = time.time()
            model.zero_grad()       
            for data in train_data:
                seq_lens,data_id,(doc_ids,pairs),labels,sents,poss,fts,revs,lidx_start,lidx_end,ridx_start,ridx_end,_ = togpu_data(data)
                idx_u = []
                idx_u_r = []
                idx_c = []
                idx_c_r = []
                idx_l = []
                idx_l_r = []
                for i, ids in enumerate(data_id):
                    if ids[0] == 'C':
                        if revs[i]:
                            idx_c_r.append(i)
                        else:
                            idx_c.append(i)
                    elif ids[0] == 'L':
                        if revs[i]:
                            idx_l_r.append(i)
                        else:
                            idx_l.append(i)
                if len(idx_l) > 0:
                    seq_l = seq_lens[idx_l]
                    sent = sents[idx_l]
                    pos = poss[idx_l]
                    ft = fts[idx_l]
                    l_start = lidx_start[idx_l]
                    l_end = lidx_end[idx_l]
                    r_start = ridx_start[idx_l]
                    r_end = ridx_end[idx_l]
                    out, prob = model(seq_l, (sent, pos, ft), l_start, l_end, 
                                      r_start, r_end, flip=False, causal=False)
                    label = labels[idx_l]
                    doc_id = [doc_ids[i] for i in idx_l]
                    pair = [pairs[i] for i in idx_l]
                    for i in range(len(doc_id)):
                        left = pair[i][0]
                        right = pair[i][1]
                        train_pairs.append(("%s_%s"%(doc_id[i], left), "%s_%s"%(doc_id[i], right)))
                    probs.append(prob)
                    gt_labels.append(label)

                if len(idx_l_r) > 0:
                    seq_l = seq_lens[idx_l_r]
                    sent = sents[idx_l_r]
                    pos = poss[idx_l_r]
                    ft = fts[idx_l_r]
                    l_start = lidx_start[idx_l_r]
                    l_end = lidx_end[idx_l_r]
                    r_start = ridx_start[idx_l_r]
                    r_end = ridx_end[idx_l_r]
                    out, prob = model(seq_l, (sent, pos, ft), l_start, l_end, 
                                      r_start, r_end, flip=True, causal=False)
                    label = labels[idx_l_r]
                    doc_id = [doc_ids[i] for i in idx_l_r]
                    pair = [pairs[i] for i in idx_l_r]
                    for i in range(len(doc_id)):
                        left = pair[i][0]
                        right = pair[i][1]
                        train_pairs_r.append(("%s_%s"%(doc_id[i], right), "%s_%s"%(doc_id[i], left)))
                    probs_r.append(prob)
                    gt_labels_r.append(label)
                    
                if (len(idx_c) > 0) and args.joint:
                    seq_l = seq_lens[idx_c]
                    sent = sents[idx_c]
                    pos = poss[idx_c]
                    ft = fts[idx_c]
                    l_start = lidx_start[idx_c]
                    l_end = lidx_end[idx_c]
                    r_start = ridx_start[idx_c]
                    r_end = ridx_end[idx_c]
                    out, prob = model(seq_l, (sent, pos, ft), l_start, l_end, 
                                      r_start, r_end, flip=False, causal=True)
                    label = labels[idx_c]
                    predicted = (prob.data.max(1)[1]).long().view(-1)
                    correct += (predicted == label.data).sum()
                    doc_id = [doc_ids[i] for i in idx_c]
                    pair = [pairs[i] for i in idx_c]
                    for i in range(len(doc_id)):
                        left = pair[i][0]
                        right = pair[i][1]
                        train_pairs_c.append(("%s_%s"%(doc_id[i], left), "%s_%s"%(doc_id[i], right)))
                    probs_c.append(prob)
                    gt_labels_c.append(label)
                
                if (len(idx_c_r) > 0) and args.joint:
                    seq_l = seq_lens[idx_c_r]
                    sent = sents[idx_c_r]
                    pos = poss[idx_c_r]
                    ft = fts[idx_c_r]
                    l_start = lidx_start[idx_c_r]
                    l_end = lidx_end[idx_c_r]
                    r_start = ridx_start[idx_c_r]
                    r_end = ridx_end[idx_c_r]
                    out, prob = model(seq_l, (sent, pos, ft), l_start, l_end, 
                                      r_start, r_end, flip=True, causal=True)
                    label = labels[idx_c_r]
                    predicted = (prob.data.max(1)[1]).long().view(-1)
                    correct += (predicted == label.data).sum()
                    doc_id = [doc_ids[i] for i in idx_c_r]
                    pair = [pairs[i] for i in idx_c_r]
                    for i in range(len(doc_id)):
                        left = pair[i][0]
                        right = pair[i][1]
                        train_pairs_c_r.append(("%s_%s"%(doc_id[i], right), "%s_%s"%(doc_id[i], left)))
                    probs_c_r.append(prob)
                    gt_labels_c_r.append(label)
                step += 1 
            # perform global inference
            # concat all data first
            train_pairs = train_pairs+train_pairs_r
            train_pairs_c = train_pairs_c+train_pairs_c_r
            probs = torch.cat((probs+probs_r), dim=0)
            prob_table = probs.cpu().data.numpy()
            gt_labels = torch.cat(gt_labels+gt_labels_r, dim=0)
            prob_table_c = np.zeros((0, 0))
            if len(probs_c) > 0:
                probs_c = torch.cat((probs_c+probs_c_r), dim=0)
                prob_table_c = probs_c.cpu().data.numpy()
                gt_labels_c = torch.cat(gt_labels_c+gt_labels_c_r, dim=0)
            
            # find max prediction based on global prediction 
            best_pred_idx, best_pred_idx_c =\
                self.global_prediction(train_pairs, prob_table, train_pairs_c,
                                       prob_table_c, backward=(args.trainon!='forward'),
                                       trans_only=(args.trans_only))
            
            loss = self.loss_func(best_pred_idx, gt_labels, probs, args.margin)
            if len(probs_c) > 0:
                loss_c = self.loss_func(best_pred_idx_c, gt_labels_c, probs_c, args.margin)
                loss = loss + loss_c
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()                               
            if not in_cv:
                print("Train loss: %.4f" % loss.cpu().item())
            ###### Evaluate at the end of each epoch ##### 
            if len(eval_data) > 0:
                eval_gt, eval_preds = self.predict(model, eval_data, args, in_dev=True)
                eval_f1 = self.weighted_f1(eval_preds, eval_gt)
                #ta_f1 = temporal_awareness(eval_data, [self._id_to_label[x] for x in eval_preds])
                if eval_f1 > best_eval_f1:
                    best_eval_f1 = eval_f1
                    if not in_cv:
                        print('Save model in %s epoch' %(epoch+1))
                        print('Best Evaluation F1: %.4f' %(eval_f1))
                        self.model = copy.deepcopy(model)
                    best_epoch = epoch + 1
                    early_stop_counter=0
                if early_stop_counter >= args.earlystop:
                    break
                else:
                    early_stop_counter += 1
                print("Evaluation F1: %.4f" % eval_f1)
                print("*"*50)
        print("Final Evaluation F1: %.4f" % best_eval_f1)
        print("*"*50)
        if len(eval_data) == 0 or (args.epochs==0):
            self.model = copy.deepcopy(model)
            best_epoch = args.epochs
        
        if args.save_model and (not in_cv):
            torch.save({'args': args,
                        'state_dict': self.model.state_dict(),
                        'f1': best_eval_f1,
                        'optimizer' : optimizer.state_dict(),
                        'epoch': best_epoch
                        },"%sglobal_best_%s.pt" % (args.ilp_dir, args.save_stamp))
        
        return best_eval_f1, best_epoch
    
    def loss_func(self, best_pred_idx, gt_labels, probs, margin):
        mask_pred = togpu(torch.ByteTensor(best_pred_idx))
        assert mask_pred.size() == probs.size()
        max_scores = torch.masked_select(probs, mask_pred) # S(y^;x) ; 1D array
        # true label scores
        N = probs.size()[0]
        C = probs.size()[1]
        assert max_scores.size(0) == N
        idx_mat = np.zeros((N, C), dtype=int)
        for n in range(N):
            idx_mat[n][gt_labels[n]] = 1
        mask = togpu(torch.ByteTensor(idx_mat))
        assert mask.size() == probs.size()
        label_scores = torch.masked_select(probs, mask) # S(y;x) ; 1D array
        if margin == 0.0:
            # Hammming distance
            delta = torch.sum((mask_pred!=mask), dim=1, dtype=torch.float)
        else:
            delta = togpu(torch.FloatTensor([margin for n in range(N)]))
        diff = 0.1*delta + (max_scores - label_scores) # size N
        mask = (diff<0.0)
        loss_t = diff.masked_fill_(mask, 0.0)
        losses = torch.mean(loss_t)
        return losses
    
    def global_prediction(self, pairs, prob_table, pairs_c, prob_table_c, 
                          evaluate=False, true_labels=[], backward=True, trans_only=False):
        # input:                                            
        # 1. pairs: doc_id + entity_id     
        # 2. prob_table: numpy matrix of local predictions (N * C)
        # 3. evaluate: True - print classification report
        # 4. true_label: if evaluate is true, need to true_label to evaluate model
        # output:                                      
        # 1. if evaluate, print classification report and return best global assignment  
        # 2. else, class selection for each sample store in matrix form                   
        N, C = prob_table.shape
        Nc, Cc = prob_table_c.shape
        global_model = Gurobi_Inference(pairs, prob_table, pairs_c, prob_table_c, 
                                        self._label_to_id, self._label_to_id_c, 
                                        backward=backward, trans_only=trans_only)
        global_model.run()
        global_model.predict()
        best_pred_idx = np.zeros((N, C), dtype=int)
        best_pred_idx_c = np.zeros((Nc, Cc), dtype=int)
        # temporal
        for n in range(N):
            best_pred_idx[n, global_model.pred_labels[n]] = 1
        # causal
        for n in range(Nc):
            best_pred_idx_c[n, global_model.pred_labels_c[n]] = 1
        
        if evaluate:
            assert len(true_labels) == N + Nc
            if self.args.data_type == 'tbd': 
                prediction = global_model.evaluate(true_labels, exclude_vague=False)
            else:
                prediction = global_model.evaluate(true_labels, exclude_vague=True)
            prediction = togpu(torch.LongTensor(prediction))
            return best_pred_idx, best_pred_idx_c, prediction
        else:
            return best_pred_idx, best_pred_idx_c

    def cross_validation(self, emb, pos_emb, args):
        param_perf = []
        for param in ParameterGrid(args.params):
            param_str = ""
            for k,v in param.items():
                param_str += "%s=%s" % (k, v)
                param_str += " "
            print("*" * 50)
            print("Train parameters: %s" % param_str)

            for k,v in param.items():
                exec("args.%s=%s" % (k, v))
            all_splits = [x for x in range(args.n_splits)]
            if (not args.cuda) or (not torch.cuda.is_available()):            
                with mp.Pool(processes=args.n_splits) as pool:
                    res = pool.map(partial(self.parallel_cv, emb=emb, 
                                           pos_emb=pos_emb, args=args), all_splits)
            else:
                res = []
                for split in all_splits:
                    res.append(self.parallel_cv(split, emb=emb, pos_emb=pos_emb, args=args))
            f1s = list(zip(*res))[0]
            best_epoch = list(zip(*res))[1]
            print('avg f1 score: %s, avg epoch: %s'%(np.mean(f1s),np.mean(best_epoch)))
            param_perf.append((param, np.mean(f1s), np.mean(best_epoch)))
            if args.write:
                with open('best_param/global_cv_devResult_'+str(args.data_type)+
                          '_TrainOn'+str(args.trainon)+'_TestOn'+str(args.teston)+
                          '_uf'+str(args.usefeature)+'_trainpos'+str(args.train_pos_emb)+
                          '_joint'+str(args.joint)+'_devbytrain'+str(args.devbytrain)+
                          '.pickle', 'wb') as f:
                    pickle.dump(sorted(param_perf, key=lambda x:x[1], reverse=True), 
                                f, pickle.HIGHEST_PROTOCOL)
        params, f1, epoch = sorted(param_perf, key=lambda x: x[1], reverse=True)[0]
        print('*' * 50)
        print("Best Average F1: %s" % f1)
        print("Best Parameters Are: %s " % params)
        print("Best Epoch is: %s" % epoch)
        print('*' * 50)
        return params, epoch

    def selectparam(self, emb, pos_emb, args):
        param_perf = []
        for param in ParameterGrid(args.params):
            param_str = ""
            for k,v in param.items():
                param_str += "%s=%s" % (k, v)
                param_str += " "
            print("*" * 50)
            print("Train parameters: %s" % param_str)
            for k,v in param.items():
                exec("args.%s=%s" % (k, v))
            if (not args.cuda) or (not torch.cuda.is_available()):    
                raise ValueError('cuda not available')
                pass #TODO
            else:
                f1, best_epoch = self.parallel_selectparam(emb=emb, pos_emb=pos_emb, args=args)
                print('param f1 score: %s, param epoch %s'%(f1, best_epoch))
            param_perf.append((param, f1, best_epoch))
            if args.write:
                with open('best_param/global_selectparam_devResult_'+str(args.data_type)+
                          '_TrainOn'+str(args.trainon)+'_TestOn'+str(args.teston)+
                          '_uf'+str(args.usefeature)+'_trainpos'+str(args.train_pos_emb)+
                          '_joint'+str(args.joint)+'_devbytrain'+str(args.devbytrain)+
                          '.pickle', 'wb') as f:
                    pickle.dump(sorted(param_perf, key=lambda x:x[1], reverse=True), 
                                f, pickle.HIGHEST_PROTOCOL)
        params, f1, epoch = sorted(param_perf, key=lambda x: x[1], reverse=True)[0]
        print('*' * 50)
        print("Best Dev F1: %s" % f1)
        print("Best Parameters Are: %s " % params)
        print("Best Epoch is: %s" % epoch)
        print('*' * 50)
        return params, epoch

    def parallel_selectparam(self, emb = np.array([]), pos_emb = [], args=None):
        params = {'batch_size': args.batch,
                  'shuffle': False}
        if args.bert_fts:
            type_dir = "all_bertemb/"
        else:
            type_dir = "all/"
        backward_dir = ""
        if (args.trainon=='bothway') or (args.trainon=='bothWselect'):
            if args.bert_fts:
                backward_dir = args.data_dir + "all_backward_bertemb/"
            else:
                backward_dir = args.data_dir + "all_backward/"

        train_data = EventDataset(args.data_dir+type_dir,"train",
                                  args.glove2vocab, backward_dir, args.bert_fts)
        train_generator = get_data_loader(train_data, **params)
        dev_data = EventDataset(args.data_dir+type_dir,"dev",
                                args.glove2vocab, backward_dir, args.bert_fts)
        dev_generator = get_data_loader(dev_data, **params)
        seeds = [0, 10, 20]
        accumu_f1 = 0.
        accumu_epoch = 0.
        for seed in seeds:
            exec('args.%s=%s' %('seed', seed))
            f1, epoch = self._train(train_generator, dev_generator, emb, pos_emb, args, in_cv=True)
            accumu_f1 += f1
            accumu_epoch += epoch
        avg_f1 = accumu_f1/float(len(seeds))
        avg_epoch = accumu_epoch/float(len(seeds))
        return avg_f1, avg_epoch

    def parallel_cv(self, split, emb = np.array([]), pos_emb = [], args=None):
        params = {'batch_size': args.batch,
                  'shuffle': False}
        if args.bert_fts:
            type_dir = "cv_bertemb"
        else:
            type_dir = "cv_shuffle" if args.cv_shuffle else 'cv'
        
        backward_dir = ""
        if (args.trainon=='bothway') or (args.trainon=='bothWselect'):
            if args.bert_fts:
                backward_dir = "%scv_backward_bertemb/fold%s/" % (args.data_dir, split)
            else:
                backward_dir = "%scv_backward/fold%s/" % (args.data_dir, split)
        train_data = EventDataset(args.data_dir+'%s/fold%s/'%(type_dir,split),"train",
                                  args.glove2vocab,backward_dir,args.bert_fts)
        train_generator = get_data_loader(train_data, **params)

        dev_data = EventDataset(args.data_dir+'%s/fold%s/'%(type_dir, split),"dev",
                                args.glove2vocab,backward_dir,args.bert_fts)
        dev_generator = get_data_loader(dev_data, **params)
        seeds = [0, 10, 20]
        accumu_f1 = 0.
        accumu_epoch = 0.
        for seed in seeds:
            exec('args.%s=%s' %('seed', seed))
            f1, epoch = self._train(train_generator, dev_generator, emb, pos_emb, args, in_cv=True)
            accumu_f1 += f1
            accumu_epoch += epoch
        avg_f1 = accumu_f1/float(len(seeds))
        avg_epoch = accumu_epoch/float(len(seeds))
        return avg_f1, avg_epoch
                  
    def train_epoch(self, train_data, dev_data, args, test_data = None):
        if args.data_type == "matres":
            label_map = matres_label_map
        elif args.data_type == "tbd":
            label_map = tbd_label_map
        else:
            label_map = new_label_map
        all_labels = list(OrderedDict.fromkeys(label_map.values()))
        self._label_to_id = OrderedDict([(all_labels[l],l) for l in range(len(all_labels))])
        self._id_to_label = OrderedDict([(l,all_labels[l]) for l in range(len(all_labels))])
        args.label_to_id = self._label_to_id
        if args.joint:
            label_map_c = causal_label_map
            all_labels_c =  list(OrderedDict.fromkeys(label_map_c.values()))
            self._label_to_id_c = OrderedDict([(all_labels_c[l],l) for l in range(len(all_labels_c))])
            self._id_to_label_c = OrderedDict([(l,all_labels_c[l]) for l in range(len(all_labels_c))])
        
        emb = args.emb_array
        np.random.seed(0)
        emb = np.vstack((np.random.uniform(0, 1, (2, emb.shape[1])), emb))
        assert emb.shape[0] == len(args.glove2vocab)
        pos_emb= np.zeros((len(args.pos2idx) + 2, len(args.pos2idx) + 2))
        for i in range(pos_emb.shape[0]):
            pos_emb[i, i] = 1.0
        
        self.args = args
        selected_epoch = 20
        if args.cv == True:
            best_params, avg_epoch = self.cross_validation(emb, pos_emb, copy.deepcopy(args))
            for k,v in best_params.items():
                exec("args.%s=%s" % (k, v))
            if args.write:
                with open('best_param/global_cv_bestparam_'+str(args.data_type)+
                          '_TrainOn'+str(args.trainon)+'_Teston'+str(args.teston)+
                          '_uf'+str(args.usefeature)+
                          '_trainpos'+str(args.train_pos_emb)+
                          '_joint'+str(args.joint)+
                          '_devbytrain'+str(args.devbytrain), 'w') as file:
                    for k,v in vars(args).items():
                        if (k!='emb_array') and (k!='glove2vocab'):
                            file.write(str(k)+'    '+str(v)+'\n')
            selected_epoch = avg_epoch
        elif args.selectparam ==True:
            best_params, best_epoch = self.selectparam(emb, pos_emb, args)
            for k,v in best_params.items():
                exec("args.%s=%s" % (k, v))
            if args.write:
                with open('best_param/global_selectDev_bestparam_'+
                          str(args.data_type)+'_TrainOn'+str(args.trainon)+
                          '_Teston'+str(args.teston)+
                          '_uf'+str(args.usefeature)+
                          '_trainpos'+str(args.train_pos_emb)+
                          '_joint'+str(args.joint)+
                          '_devbytrain'+str(args.devbytrain), 'w') as file:
                    for k,v in vars(args).items():
                        if (k!='emb_array') and (k!='glove2vocab'):
                            file.write(str(k)+'    '+str(v)+'\n')
            selected_epoch = best_epoch

        if args.refit_all:
            exec('args.epochs=%s'%int(selected_epoch))
            print('refit all.....')
            params = {'batch_size': args.batch,
                      'shuffle': False}
            if args.bert_fts:
                type_dir = "all_bertemb/"
            else:
                type_dir = 'all/'
            data_dir_back = ""
            if (args.trainon=='bothway') or (args.trainon=='bothWselect'):
                if args.bert_fts:
                    data_dir_back = args.data_dir + "all_backward_bertemb/"
                else:
                    data_dir_back = args.data_dir + "all_backward/"
            t_data = EventDataset(args.data_dir+type_dir,'train',args.glove2vocab,data_dir_back,args.bert_fts)
            d_data = EventDataset(args.data_dir+type_dir,'dev',args.glove2vocab,data_dir_back,args.bert_fts)
            t_data.merge_dataset(d_data)
            train_data = get_data_loader(t_data, **params)
            dev_data = []
        best_f1, best_epoch = self._train(train_data, dev_data, emb, pos_emb, args)
        print("Final Epoch Use: %s" % best_epoch)
        print("Final Dev F1: %.4f" % best_f1)
        return best_f1

    def weighted_f1(self, pred_labels, true_labels):
        def safe_division(numr, denr, on_err=0.0):
            return on_err if float(denr) == 0.0 else float(numr) / float(denr)
        assert len(pred_labels) == len(true_labels)
        if 'NONE' in self._label_to_id.keys():
            num_tests = len([x for x in true_labels if x != self._label_to_id['NONE']])
        else:
            num_tests = len([x for x in true_labels])

        total_true = Counter(true_labels)
        total_pred = Counter(pred_labels)
        labels = list(self._id_to_label.keys())
        n_correct = 0
        n_true = 0
        n_pred = 0

        exclude_labels = ['NONE', 'VAGUE'] if len(self._label_to_id) == 4 else ['NONE']
        for label in labels:
            if self._id_to_label[label] not in exclude_labels:
                true_count = total_true.get(label, 0)
                pred_count = total_pred.get(label, 0)
                n_true += true_count
                n_pred += pred_count

                correct_count = 0
                for l in range(len(pred_labels)):
                    if pred_labels[l] == true_labels[l] and pred_labels[l] == label:
                        correct_count += 1
                n_correct += correct_count
        precision = safe_division(n_correct, n_pred) 
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2 * precision * recall, precision + recall)
        return f1_score

def temporal_awareness(data, pred_labels, data_type, with_timex=False):
    gold_rels = {}
    for i, ex in enumerate(data):
        # Do not evaluate on VAGUE class for MATRES dataset
        if data_type == 'matres' and ex.rel_type == 'VAGUE':
            continue
        if ex[0] in gold_rels:
            gold_rels[ex[0]].append((i, ex[1], ex[2], ex[3]))
        else:
            gold_rels[ex[0]] = [(i, ex[1], ex[2], ex[3])]

    idx2docs = {}
    for k, vs in gold_rels.items():
        for v in vs:
            idx2docs[v[0]] = (k, v[1], v[2], v[3])
    if data_type == 'tbd' and with_timex:
        print("TBDense Gold")
        '''
        with open("CAEVO_ETTT_OUTPUT", "rb") as fl:
            gold = pickle.load(fl)
            for k, v in gold.items():
                print(k, len(v))
                gold_rels[k].extend([(0, kk[0], kk[1], vv) for kk,vv in v.items()])
                print(len(gold_rels[k]))
        '''
    pred_rels = {}
    for i, pl in enumerate(pred_labels):
        try:
            if idx2docs[i][0] in pred_rels:
                pred_rels[idx2docs[i][0]].append((idx2docs[i][1], idx2docs[i][2], pl))
            else:
                pred_rels[idx2docs[i][0]] = [(idx2docs[i][1], idx2docs[i][2], pl)]
        except:
            # VAGUE pairs in matres, excluded
            continue
    if data_type == 'tbd' and with_timex:
        ### append ET and TT pairs 
        print("CAEVO Predictions")
        '''
        with open("CAEVO_ETTT_OUTPUT", "rb") as fl:
            pred = pickle.load(fl)
            for k, v in pred.items():
                pred_rels[k].extend([(kk[0], kk[1], vv) for kk,vv in v.items()])
        '''
    return evaluate_all(gold_rels, pred_rels)

def main_global(args):
    data_dir = args.data_dir
    params = {'batch_size': args.batch,
              'shuffle': False}
    if args.bert_fts:
        type_dir = "all_bertemb/"
    else:
        type_dir = "all/"
    data_dir_back = ""
    if (args.trainon=='bothway') or (args.trainon=='bothWselect'):
        if args.bert_fts:
            data_dir_back = args.data_dir + "all_backward_bertemb/"
        else:
            data_dir_back = args.data_dir + "all_backward/"
    train_data = EventDataset(args.data_dir + type_dir, "train", 
                              args.glove2vocab, data_dir_back, args.bert_fts)
    print('train_data: %s in total' % len(train_data))
    train_generator = get_data_loader(train_data, **params)
    dev_data = EventDataset(args.data_dir + type_dir, "dev", 
                            args.glove2vocab, data_dir_back, args.bert_fts)
    print('dev_data: %s in total' % len(dev_data))
    dev_generator = get_data_loader(dev_data, **params)
    
    if args.bert_fts:
        data_dir_back = args.data_dir + "all_backward_bertemb/"
    else:
        data_dir_back = args.data_dir + "all_backward/"
    test_data = EventDataset(args.data_dir + type_dir, "test", 
                             args.glove2vocab, data_dir_back, args.bert_fts)
    test_generator = get_data_loader(test_data, **params)
    
    s_time = time.time() 
    models = [NNClassifier()]
    score = 0
    for model in models:
        dev_f1 = model.train_epoch(train_generator, dev_generator, args)
        print('total time escape', time.time()-s_time)
        evaluator = Evaluator(model)
        #print(evaluator.evaluate(test_generator, args))
        score = evaluator.get_score(test_generator, args)
        #evaluator.collect_result(test_generator, args)
        print('final test f1: %.4f' %(score))
    return float(dev_f1), float(score)
