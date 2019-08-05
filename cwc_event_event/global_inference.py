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
from baseline import REDEveEveRelModel, matres_label_map, tbd_label_map, new_label_map, red_label_map, causal_label_map
from baseline import  ClassificationReport, rev_map, rev_causal_map
from featureFuncs import *
from gurobi_inference import Gurobi_Inference
import multiprocessing as mp
from functools import partial
from sklearn.model_selection import ParameterGrid
from ldctcr import NewDoc, NewRelation, NewEntity
from ldctbd import TBDDoc, TBDRelation, TBDEntity
from temporal_evaluation import *
from nn_model import BiLSTM
from dataloader import get_data_loader
from dataset import EventDataset
 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(123)

@dataclass
class REDEvaluator:
    model: REDEveEveRelModel
    def evaluate(self, test_data, args):
        true_labels, pred_labels = self.model.predict(self.model.model, test_data, args)
        print('test_data len', len(true_labels), len(pred_labels))

        pred_labels = [self.model._id_to_label[x] for x in pred_labels]
        true_labels = [self.model._id_to_label[x] for x in true_labels]

        assert len(pred_labels) == len(true_labels)

        if args.data_type in ['']:
            test_data = [(x[0][0], x[2][0][0], x[2][1][0], true_labels[k])
                         for k, x in enumerate(test_data) if k < len(true_labels)]
            temporal_awareness(test_data, pred_labels, args.data_type, args.eval_with_timex)
        
        #ids = [x[1][0] for x in test_data if x[1][0][0] != 'C']
        #self.for_analysis(ids, true_labels, pred_labels, test_data, args.ilp_dir+'matres_global_all.tsv')
        if args.data_type == 'tbd':
            return ClassificationReport(self.model.name, true_labels, pred_labels, False)
        else:
            return ClassificationReport(self.model.name, true_labels, pred_labels)

    def for_analysis(self, ids, golds, preds, test_data, outfile):
        with open(outfile, 'w') as file:
            file.write('\t'.join(['doc_id', 'pair_id', 'label', 'pred', 'left_text', 'right_text', 'context']))
            file.write('\n')
            i = 0
            i2w = np.load('i2w.npy').item()
            v2g = np.load('v2g.npy').item()
            for ex in test_data:
                left_s = ex[8].tolist()[0]
                left_e = ex[9].tolist()[0]
                right_s = ex[10].tolist()[0]
                right_e = ex[11].tolist()[0]
                context = [i2w[v2g[x]] for x in ex[4][0].tolist()]
                left_text = context[left_s : left_e + 1][0]
                right_text  = context[right_s : right_e + 1][0]
                context = ' '.join(context)
                file.write('\t'.join([ex[0][0],
                                      ids[i],
                                      golds[i],
                                      preds[i],
                                      left_text,
                                      right_text,
                                      context]))
                file.write('\n')
                i += 1
                print(i)
        file.close()
        return

@dataclass()
class NNClassifier(REDEveEveRelModel):
    label_probs: Optional[List[float]] = None
    _label_to_id_c = {}
    _id_to_label_c = {}
    permu_indices = None
    permu_indices_c = None

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
            gt_labels = torch.cat((gt_labels,gt_labels_c), dim=0)
        
        # find max prediction based on global prediction 
        best_pred_idx, best_pred_idx_c, predictions=\
            self.global_prediction(eval_pairs, prob_table, eval_pairs_c,
                                   prob_table_c, evaluate=True,
                                   true_labels=gt_labels, flip=True)
        loss = self.loss_func(best_pred_idx, gt_labels, probs, args.margin, 
                              best_pred_idx_c, probs_c)
        print("Evaluation loss: %.4f" % loss.cpu().data.numpy())
        if not in_dev:
            N = len(eval_pairs)
            if args.teston=='forward':
                final_pred_labels = predictions[:int(N/2)]
                final_gt_labels = gt_labels[:int(N/2)]
            elif args.teston=='backward':
                final_pred_labels = predictions[int(N/2):N]
                final_gt_labels = gt_labels[int(N/2):N]
            elif args.teston=='bothway':
                final_pred_labels = predictions[:N]
                final_gt_labels = gt_labels[:N]
        else:
            if (args.trainon=='forward') or (args.trainon=='strictforward'):
                final_pred_labels = predictions[:int(N/2)]
                final_gt_labels = gt_labels[:int(N/2)]
            else:
                final_pred_labels = predictions[:N]
                final_gt_labels = gt_labels[:N]

        return final_gt_labels.cpu().tolist(), final_pred_labels.cpu().tolist()
    
    def get_reverse_prob(self, ori_prob):
        if self.permu_indices is None:
            rev_id2id={}
            for i, label in self._id_to_label.items():
                rev_id2id[i] = self._label_to_id[rev_map[label]]
            permu_indices = togpu(torch.zeros((len(self._id_to_label),), dtype=torch.long))
            for id_ori, id_rev in rev_id2id.items():
                permu_indices[id_ori]=id_rev
            self.permu_indices = permu_indices
        return ori_prob[self.permu_indices]

    def get_reverse_prob_c(self, ori_prob):
        if self.permu_indices_c is None:
            rev_id2id={}
            for i, label in self._id_to_label_c.items():
                rev_id2id[i] = self._label_to_id_c[rev_causal_map[label]]
            permu_indices = togpu(torch.zeros((len(self._id_to_label_c,)), dtype=torch.long))
            for id_ori, id_rev in rev_id2id.item():
                permu_indices[id_ori]=id_rev
            self.permu_indices_c = permu_indices
        return ori_prob[self.permu_indices_c]

    def _train(self, train_data, eval_data, emb, pos_emb, args, 
               in_cv=False, test_data=None):
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
                    elif ids[0] == 'U':
                        if revs[i]:
                            idx_u_r.append(i)
                        else:
                            idx_u.append(i)
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

                if (args.skip_u) or (args.loss_u==''):
                    pass
                else:
                    if len(idx_u) > 0:
                        seq_l = seq_lens[idx_u]
                        sent = sents[idx_u]
                        pos = poss[idx_u]
                        ft = fts[idx_u]
                        l_start = lidx_start[idx_u]
                        l_end = lidx_end[idx_u]
                        r_start = ridx_start[idx_u]
                        r_end = ridx_end[idx_u]
                        out, prob = model(seq_l, (sent, pos, ft), l_start,
                                          l_end, r_start, r_end, flip=False, causal=False)
                    if len(idx_u_r) > 0:
                        seq_l = seq_lens[idx_u_r]
                        sent = sents[idx_u_r]
                        pos = poss[idx_u_r]
                        ft = fts[idx_u_r]
                        l_start = lidx_start[idx_u_r]
                        l_end = lidx_end[idx_u_r]
                        r_start = ridx_start[idx_u_r]
                        r_end = ridx_end[idx_u_r]
                        out, prob = model(seq_l, (sent, pos, ft), l_start,
                                          l_end, r_start, r_end, flip=True, causal=False)
                step += 1 

            # perform global inference
            # concat all data first
            train_pairs = train_pairs+train_pairs_r
            train_pairs_c = train_pairs_c+train_pairs_c_r
            if args.trainon=='strictforward':
                probs_r = torch.cat(probs_r, dim=0).detach()
                probs_f = torch.cat(probs, dim=0)
                probs = torch.cat((probs_f, probs_r), dim=0)
            else:
                probs = torch.cat((probs+probs_r), dim=0)
            prob_table = probs.cpu().data.numpy()
            gt_labels = torch.cat(gt_labels+gt_labels_r, dim=0)
            prob_table_c = np.zeros((0, 0))
            if len(probs_c) > 0:
                if args.trainon=='strictforward':
                    probs_c_r = torch.cat(probs_c_r, dim=0).detach()
                    probs_c_f = torch.cat(probs_c, dim=0)
                    probs_c = torch.cat((probs_c_f, probs_c_r), dim=0)
                else:
                    probs_c = torch.cat((probs_c+probs_c_r), dim=0)
                prob_table_c = probs_c.cpu().data.numpy()
                gt_labels_c = torch.cat(gt_labels_c+gt_labels_c_r, dim=0)
                gt_labels = torch.cat((gt_labels,gt_labels_c), dim=0)
            
            # find max prediction based on global prediction 
            best_pred_idx, best_pred_idx_c =\
                self.global_prediction(train_pairs, prob_table, train_pairs_c,
                                       prob_table_c, flip=True)
            
            loss = self.loss_func(best_pred_idx, gt_labels, probs, args.margin,
                                  best_pred_idx_c, probs_c)
            #if len(probs_c) > 0:
                #loss_c = self.loss_func(best_pred_idx_c, gt_labels_c, probs_c, args.margin)
                #loss = loss + loss_c
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()                               
            if not in_cv:
                print("Train loss: %.4f" % loss.cpu().item())
                print("*"*50)
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
                #print("Evaluation F1: %.4f" % eval_f1)
                #print("*"*50)
        print("Final Evaluation F1: %.4f" % best_eval_f1)
        print("*"*50)
        if len(eval_data) == 0 :
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
    
    def loss_func(self, best_pred_idx, gt_labels, probs, margin,
                  best_pred_idx_c, probs_c):
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
        if (len(probs_c)>0) and (self.args.joint):
            mask_pred_c = togpu(torch.ByteTensor(best_pred_idx_c))
            assert mask_pred_c.size() == probs_c.size()
            max_scores_c = torch.masked_select(probs_c, mask_pred_c)
            Nc = probs_c.size()[0]
            Cc = probs_c.size()[1]
            assert max_scores_c.size(0) == Nc
            idx_mat_c = np.zeros((Nc, Cc), dtype=int)
            for n in range(Nc):
                idx_mat_c[n][gt_labels[n+N]] = 1
            mask_c = togpu(torch.ByteTensor(idx_mat_c))
            assert mask_c.size() == probs_c.size()
            label_scores_c = torch.masked_select(probs_c, mask_c) # S(y;x) ; 1D array

        if margin == -1.0:
            # Hammming distance
            delta = torch.sum((mask_pred!=mask), dim=1, dtype=torch.float)
        else:
            delta = togpu(torch.FloatTensor([margin for n in range(N)]))
        diff = delta + max_scores - label_scores # size N
        mask = (diff<0.0)
        loss_t = diff.masked_fill_(mask, 0.0)
        losses = torch.mean(loss_t)
        if (len(probs_c)>0) and (self.args.joint):
            if margin == -1.0:
                # Hammming distance
                delta_c = torch.sum((mask_pred_c!=mask_c), dim=1, dtype=torch.float)
            else:
                delta_c = togpu(torch.FloatTensor([margin for n in range(Nc)]))
            diff_c = delta_c + max_scores_c - label_scores_c
            mask_c = (diff_c<0.0)
            loss_c = diff_c.masked_fill_(mask_c, 0.0)
            losses += torch.mean(loss_c)
        return losses
    
    def global_prediction(self, pairs, prob_table, pairs_c, prob_table_c, 
                          evaluate=False, true_labels=[], flip=True):
        # input:                                            
        # 1. pairs: doc_id + entity_id     
        # 2. prob_table: numpy matrix of local predictions (N * C)
        # 3. evaluate: True - print classification report
        # 4. true_label: if evaluate is true, need to true_label to evaluate model
        # output:                                      
        # 1. if evaluate, print classification report and return best global assignment  
        # 2. else, class selection for each sample store in matrix form                   
        assert flip==True                                  
        N, C = prob_table.shape
        Nc, Cc = prob_table_c.shape
        global_model = Gurobi_Inference(pairs, prob_table, pairs_c, prob_table_c, self._label_to_id, self._label_to_id_c)
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
                prediction = global_model.evaluate(true_labels, exclude_vague=False, backward=flip)
            else:
                prediction = global_model.evaluate(true_labels, exclude_vague=True, backward=flip)
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
                    res = pool.map(partial(self.parallel_cv, emb=emb, pos_emb=pos_emb, args=args), all_splits)
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
                          '_joint'+str(args.joint)+'.pickle', 'wb') as f:
                    pickle.dump(sorted(param_perf, key=lambda x:x[1], reverse=True), f, pickle.HIGHEST_PROTOCOL)
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
                          '_joint'+str(args.joint)+'.pickle', 'wb') as f:
                    pickle.dump(sorted(param_perf, key=lambda x:x[1], reverse=True), f, pickle.HIGHEST_PROTOCOL)
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
            type_dir = "all_bert_%sfts/" % args.n_fts
        else:
            type_dir = "all/"
        backward_dir = args.data_dir + "all_backward/"

        train_data = EventDataset(args.data_dir+type_dir,"train",
                                  args.glove2vocab,backward_dir)

        train_generator = get_data_loader(train_data, **params)

        dev_data = EventDataset(args.data_dir+type_dir,"dev",
                                args.glove2vocab,backward_dir)

        dev_generator = get_data_loader(dev_data, **params)

        return self._train(train_generator, dev_generator, emb, pos_emb, args, in_cv=True)

    def parallel_cv(self, split, emb = np.array([]), pos_emb = [], args=None):
        params = {'batch_size': args.batch,
                  'shuffle': False}
        if args.bert_fts:
            type_dir = "cv_bert_%sfts" % args.n_fts
        else:
            type_dir = "cv_shuffle" if args.cv_shuffle else 'cv'

        backward_dir = "%scv_backward/fold%s/" % (args.data_dir, split)

        train_data = EventDataset(args.data_dir+'%s/fold%s/'%(type_dir,split),"train",
                                  args.glove2vocab,backward_dir)
        train_generator = get_data_loader(train_data, **params)

        dev_data = EventDataset(args.data_dir+'%s/fold%s/'%(type_dir, split),"dev",
                                args.glove2vocab,backward_dir)
        dev_generator = get_data_loader(dev_data, **params)
        return self._train(train_generator, dev_generator, emb, pos_emb, args, in_cv=True)
                  
    def train_epoch(self, train_data, dev_data, args, test_data = None):
        if args.data_type == "red":
            label_map = red_label_map
        elif args.data_type == "matres":
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
        np.random.seed(args.seed)
        emb = np.vstack((np.random.uniform(0, 1, (2, emb.shape[1])), emb))
        assert emb.shape[0] == len(args.glove2vocab)
        pos_emb= np.zeros((len(args.pos2idx) + 2, len(args.pos2idx) + 2))
        for i in range(pos_emb.shape[0]):
            pos_emb[i, i] = 1.0
        
        self.args = args
        if (args.readcvresult==True) and (args.cvresultpath!=''):
            cvresult = pickle.load(open(args.cvresultpath, 'rb'))
            best_params = cvresult[0][0]
            for k,v in best_params.items():
                exec('args.%s' %(k,v))

        selected_epoch = 20
        if args.cv == True:
            best_params, avg_epoch = self.cross_validation(emb, pos_emb, copy.deepcopy(args))
            for k,v in best_params.items():
                exec("args.%s=%s" % (k, v))
            if args.write:
                with open('best_param/global_cv_bestparam_'+str(args.data_type)+
                          '_TrainOn'+str(args.trainon)+'_Teston'+str(args.teston)+
                          '_joint'+str(args.joint), 'w') as file:
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
                          '_Teston'+str(args.teston)+'_joint'+str(args.joint), 'w') as file:
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
                type_dir = 'all_bert_%sfts/'%args.n_fts
            else:
                type_dir = 'all/'
            data_dir_back = args.data_dir + 'all_backward/'
            t_data = EventDataset(args.data_dir+type_dir,'train',args.glove2vocab,data_dir_back)
            d_data = EventDataset(args.data_dir+type_dir,'dev',args.glove2vocab,data_dir_back)
            t_data.merge_dataset(d_data)
            train_data = get_data_loader(t_data, **params)
            dev_data = []
        best_f1, best_epoch = self._train(train_data, dev_data, emb, pos_emb, args)
        print("Final Epoch Use: %s" % best_epoch)
        print("Final Dev F1: %.4f" % best_f1)
        return

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
        with open("/nas/home/rujunhan/data/TBDense/caevo_test_ettt.pkl", "rb") as fl:
            gold = pickle.load(fl)
            for k, v in gold.items():
                print(k, len(v))
                #gold_rels[k] = [(0, kk[0], kk[1], vv) for kk,vv in v.items()]
                gold_rels[k].extend([(0, kk[0], kk[1], vv) for kk,vv in v.items()])
                print(len(gold_rels[k]))

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
        with open("/nas/home/rujunhan/CAEVO/caevo_test_ettt.pkl", "rb") as fl:
            pred = pickle.load(fl)
            for k, v in pred.items():
                #pred_rels[k] = [(kk[0], kk[1], vv) for kk,vv in v.items()]
                pred_rels[k].extend([(kk[0], kk[1], vv) for kk,vv in v.items()])
    
    return evaluate_all(gold_rels, pred_rels)

def main(args):
    data_dir = args.data_dir
    params = {'batch_size': args.batch,
              'shuffle': False}
    if args.bert_fts:
        type_dir = "all_bert_%sfts/" % args.n_fts
    else:
        type_dir = "all/"
    data_dir_back = args.data_dir + "all_backward/"
    train_data = EventDataset(args.data_dir + type_dir, "train", args.glove2vocab, data_dir_back)
    print('train_data: %s in total' % len(train_data))
    train_generator = get_data_loader(train_data, **params)
    dev_data = EventDataset(args.data_dir + type_dir, "dev", args.glove2vocab, data_dir_back)
    print('dev_data: %s in total' % len(dev_data))
    dev_generator = get_data_loader(dev_data, **params)
    
    test_data = EventDataset(args.data_dir + type_dir, "test", args.glove2vocab, data_dir_back)
    print('test_data: %s in total' % len(test_data))
    test_generator = get_data_loader(test_data, **params)
    s_time = time.time() 
    models = [NNClassifier()]
    for model in models:
        print(f"\n======={model.name}=====")
        if args.bootstrap:
            model.train_epoch(train_generator, dev_generator, args, test_data = test_generator)
            print("Finished Bootstrap Testing")
        else:
            model.train_epoch(train_generator, dev_generator, args)
            print('total time escape', time.time()-s_time)
            evaluator = REDEvaluator(model)
            print(evaluator.evaluate(test_generator, args))
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

if __name__ == '__main__':

    p = argparse.ArgumentParser()
    p.add_argument('-data_type', type=str, default="matres")
    p.add_argument('-emb', type=int, default=300)
    p.add_argument('-hid', type=int, default=60)
    p.add_argument('-num_layers', type=int, default=1)
    p.add_argument('-dropout', type=float, default=0.4)
    p.add_argument('-joint', type=str2bool, default=False)
    p.add_argument('-num_causal', type=int, default=2)
    p.add_argument('-batch', type=int, default=64)
    p.add_argument('-epochs', type=int, default=20)
    p.add_argument('-optimizer',  type=str, choices=['Adam', 'SGD'],
                   default='SGD')
    p.add_argument('-seed', type=int, default=9)
    p.add_argument('-lr', type=float, default=0.0005)
    p.add_argument('-decay', type=float, default=1e-6)
    p.add_argument('-momentum', type=float, default=0.9)
    p.add_argument('-attention', type=str2bool, default=False)
    p.add_argument('-usefeature', type=str2bool, default=False)
    p.add_argument('-sparse_emb', type=str2bool, default=False)
    p.add_argument('-train_pos_emb', type=str2bool, default=True)
    p.add_argument('-earlystop', type=int, default=8)
    p.add_argument('-trainon', type=str, default='forward',
                   choices=['strictforward', 'forward', 'bothway', 'bothWselect'])
    p.add_argument('-teston', type=str, default='forward',
                   choices=['forward', 'bothway', 'backward'])
    
    p.add_argument('-bert_fts', type=str2bool, default=False)
    p.add_argument('-n_fts', type=int, default=15)
    p.add_argument('-bootstrap', type=str2bool, default=False)
    p.add_argument('-loss_u', type=str, default='')
    p.add_argument('-unlabeled_weight', type=float, default=0.0)
    p.add_argument('-skip_u', type=str2bool, default=True)
    p.add_argument('-n_splits', type=int, default=5)
    p.add_argument('-cuda', type=str2bool, default=True)
    p.add_argument('-cv', type=str2bool, default=False)
    p.add_argument('-selectparam', type=str2bool, default=False)
    p.add_argument('-cv_shuffle', type=str2bool, default=False)
    p.add_argument('-refit_all', type=str2bool, default=False)
    p.add_argument('-readcvresult', type=str2bool, default=True)
    p.add_argument('--cvresultpath', type=str, default='')
    
    p.add_argument('-save_model', type=str2bool, default=True)
    p.add_argument('--save_stamp', type=str, default="matres_global_testing")
    p.add_argument('-ilp_dir', type=str, default="../ILP/")
    p.add_argument('-load_model', type=str2bool, default=True)
    p.add_argument('--load_model_file', type=str, 
                   default='matres_local_best_backTrue_trainposTrue_refitFalse.pt')
    p.add_argument('--margin', type=float, default=0.05)
    p.add_argument('-write', type=str2bool, default=False)
    args = p.parse_args()
    print(args)
    '''
    p.add_argument('-bootstrap', type=bool, default=False)
    p.add_argument('-bs_list', type=list, default=list(range(0, 5)))
    p.add_argument('-seed', type=int, default=9) # 9, 1, 10, 100, 200, 1000
    p.add_argument('-use_grammar', type=bool, default=False)
    '''
    if args.data_type == "red":
        args.data_dir = "../output_data/red_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args.data_dir, 'r')]
    elif args.data_type == "new":
        args.data_dir = "../output_data/tcr_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')] 
    elif args.data_type == "matres":
        args.data_dir = "../output_data/matres_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args.data_dir, 'r')]
    elif args.data_type == "tbd":
        args.data_dir = "../output_data/tbd_output/"
        args.train_docs = [x.strip() for x in open("%strain_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args.data_dir, 'r')]

    tags = open("../output_data/tcr_output/pos_tags.txt")
    pos2idx = {}
    idx = 0
    for tag in tags:
        tag = tag.strip()
        pos2idx[tag] = idx
        idx += 1
    args.pos2idx = pos2idx
    args.emb_array = np.load(args.data_dir + 'all' + '/emb_reduced.npy', allow_pickle=True)
    args.glove2vocab = np.load(args.data_dir + 'all' + '/glove2vocab.npy', allow_pickle=True).item()
    args.nr = 0.0
    args.tempo_filter = True
    args.skip_other = True
    args.params = {'lr': [0.1,0.01,0.05,0.001,0.005,0.0001], 
                   'momentum':[0.4, 0.7, 0.9], 
                   'decay':[0.4, 0.7, 0.9, 0.1]}
    main(args)
