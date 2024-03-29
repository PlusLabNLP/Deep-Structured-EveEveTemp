from pathlib import Path
import pickle
import argparse
from collections import Counter, OrderedDict
from dataclasses import dataclass
import numpy as np
import random
from random import shuffle
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
import copy
from base import EveEveRelModel, ClassificationReport, matres_label_map, tbd_label_map, new_label_map, rev_map, causal_label_map
from featureFuncs import *
from nn_model import BiLSTM
import multiprocessing as mp
from functools import partial
from sklearn.model_selection import ParameterGrid
from global_inference import temporal_awareness
from temporal_evaluation import *
from dataloader import get_data_loader
from dataset import EventDataset

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@dataclass
class Evaluator:
    model: EveEveRelModel
    def evaluate(self, test_data, args):
        print("start testing...")
        preds, loss, true_labels = self.model.predict(self.model.model, test_data, args, in_dev=False)
        pred_labels = [self.model._id_to_label[x] for x in preds.cpu().tolist()]
        true_labels = [self.model._id_to_label[x] for x in true_labels.cpu().tolist()]

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
        preds, loss, true_labels = self.model.predict(self.model.model, test_data, args, in_dev=False)
        preds = preds.cpu().tolist()
        true_labels = true_labels.cpu().tolist()
        return self.model.weighted_f1(preds, true_labels)[0]

    def for_analysis(self, test_data, args):
        preds, loss, true_labels = self.model.predict(self.model.model, test_data, args, in_dev=False)
        pred_labels = [self.model._id_to_label[x] for x in preds.cpu().tolist()]
        true_labels = [self.model._id_to_label[x] for x in true_labels.cpu().tolist()]
        sample_list = pickle.load(open('error_analysis_tbd_sample50', 'rb'))
        glove_emb = open('../output_data/glove.6B.300d.txt', 'r', encoding='utf-8')
        emb_dict = OrderedDict([(x.strip().split(' ')[0], [float(xx) for xx in x.strip().split(' ')[1:]]) for x in glove_emb])
        vocab = np.array(['<pad>', '<unk>'] + list(emb_dict.keys()))
        del glove_emb
        del emb_dict
        dataid2gloveid = {v:k for k,v in args.glove2vocab.items()}
        with open('tbd_sample_text', 'w') as file:
            file.write('\t\t'.join(['doc_id', 'pair_id', 'label', 'left_text', 'right_text', 'context']))
            file.write('\n')
            idx = 0
            for x in test_data:
                for i in range(x[0].size(0)):
                    if args.teston=='forward':
                        if (x[7][i]==False) and (x[1][i][0]=='L'):
                            name = str(x[2][0][i])+'_'+str(x[2][1][i][0])+'_'+str(x[2][1][i][1])
                            if name in sample_list:
                                doc_id = x[2][0][i]
                                pair_id = str(x[2][1][i])
                                label = true_labels[idx]
                                context = [vocab[dataid2gloveid[w]] for w in x[4][i].cpu().tolist()]
                                left_s = x[8][i].cpu().tolist()
                                left_e = x[9][i].cpu().tolist()
                                right_s = x[10][i].cpu().tolist()
                                right_e = x[11][i].cpu().tolist()
                                left_text = context[left_s:left_e+1]
                                left_text = ' '.join(left_text)
                                right_text  = context[right_s:right_e+1]
                                right_text = ' '.join(right_text)
                                c = ""
                                for con in context:
                                    if con !='<pad>':
                                        c = c+" "+con
                                file.write('\t\t'.join([doc_id, pair_id, label,
                                                      left_text, right_text, c]))
                                file.write('\n')
                            idx+=1
        file.close()
        return

    def collect_result(self, test_data, args):
        # collect results that used for McNemar Test
        preds, loss, true_labels = self.model.predict(self.model.model, test_data, args, in_dev=False)
        pred_labels = [self.model._id_to_label[x] for x in preds.cpu().tolist()]
        true_labels = [self.model._id_to_label[x] for x in true_labels.cpu().tolist()]
        matrix = {}
        idx = 0
        for x in test_data:
            for i in range(x[0].size(0)):
                if args.teston=='forward':
                    if (x[7][i]==False) and (x[1][i][0]=='L'):
                        correctness = (pred_labels[idx]==true_labels[idx])
                        name = 'local_seed'+str(args.seed)+'_'+str(x[2][0][i])+'_'+str(x[2][1][i][0])+'_'+str(x[2][1][i][1])
                        matrix[name]=correctness
                        idx+=1
        assert(idx==len(pred_labels))
        filename = 'local_'+args.data_type+'_seed'+str(args.seed)+'.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(matrix, f)
        return

@dataclass()
class NNClassifier(EveEveRelModel):
    def train_epoch(self, train_data, dev_data, args):
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
        emb = np.vstack((np.random.uniform(0, 1, (2, emb.shape[1])), emb)) # add 0 for pad, 1 for unk
        assert emb.shape[0] == len(args.glove2vocab)
        pos_emb= np.zeros((len(args.pos2idx) + 2, len(args.pos2idx) + 2)) # add 2, for <unk> and <pad>
        for i in range(pos_emb.shape[0]):
            pos_emb[i, i] = 1.0

        selected_epoch = args.epochs
        if args.cv == True:
            best_params, avg_epoch = self.cross_validation(emb, pos_emb, copy.deepcopy(args))
            for k,v in best_params.items():
                exec("args.%s=%s" % (k, v))
            if args.write:
                with open('best_param/cv_bestparam_'+str(args.data_type)+
                          '_TrainOn'+str(args.trainon)+'_TestOn'+str(args.teston)+
                          '_uf'+str(args.usefeature)+
                          '_trainpos'+str(args.train_pos_emb)+'_joint'+
                          str(args.joint)+'_devbytrain'+str(args.devbytrain), 'w') as file:
                    for k,v in vars(args).items():
                        if (k!='emb_array') and (k!='glove2vocab'):
                          file.write(str(k)+'   '+str(v)+'\n')
            selected_epoch = avg_epoch
        elif args.selectparam == True:
            best_params, best_epoch = self.selectparam(emb, pos_emb, copy.deepcopy(args))
            for k,v in best_params.items():
                exec("args.%s=%s" % (k,v))
            if args.write:
                with open('best_param/selectDev_bestparam_'+str(args.data_type)+
                          '_TrainOn'+str(args.trainon)+'_TestOn'+str(args.teston)+
                          '_uf'+str(args.usefeature)+
                          '_trainpos'+str(args.train_pos_emb)+'_joint'+
                          str(args.joint)+'_devbytrain'+
                          str(args.devbytrain), 'w') as file:
                    for k,v in vars(args).items():
                        if (k!='emb_array') and (k!='glove2vocab'):
                          file.write(str(k)+'   '+str(v)+'\n')
            selected_epoch = best_epoch

        if args.refit_all:
            exec('args.epochs=%s'%int(selected_epoch))
            print('refit all....')
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
            t_data = EventDataset(args.data_dir+type_dir,"train",args.glove2vocab,data_dir_back,args.bert_fts)
            print('total train_data %s samples' %len(t_data))
            d_data = EventDataset(args.data_dir+type_dir,"dev",args.glove2vocab,data_dir_back,args.bert_fts)
            print('total dev_data %s samples' %len(d_data))
            t_data.merge_dataset(d_data)
            print('total refit_data %s samples' %len(t_data))
            train_data = get_data_loader(t_data, **params)
            dev_data = []
        
        best_f1, _ = self._train(train_data, dev_data, emb, pos_emb, args)
        print("Final Dev F1: %.4f" % best_f1)
        return best_f1

    def predict(self, model, data, args, in_dev=False):
        '''
        We process predict all data no matter it's backward or forward at first.
        But we calculate scores based on argument choice.

        If trainon bothWselect:
        The idea is: assume we want to predict the relation between (a,b)
        we can put (a,b) to the classifier or put (b,a) in the classifier.
        We will consider both this situation when evaluation, and
        choose the higher probability prediction among the two to be the final prediction.
        '''
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction='none')
        step = 1
        correct = 0
        gt_labels = torch.zeros((0,), dtype=torch.long)
        probs = torch.zeros((0, len(args.label_to_id)))
        losses_t = torch.zeros((0,))
        losses_c = torch.zeros((0,))
        gt_labels_rev = torch.zeros((0,), dtype=torch.long)
        probs_rev = torch.zeros((0, len(args.label_to_id)))
        
        rev_count = 0
        for d in data:
            seq_lens,data_id,_,labels,sents,poss,fts,revs,lidx_start,lidx_end,ridx_start,ridx_end,_ = togpu_data(d)
            idx_c = []
            idx_c_r = []
            idx_t = []
            idx_t_r = []
            for i, ids in enumerate(data_id):
                if ids[0] == 'C':
                    if revs[i]:
                        idx_c_r.append(i)
                    else:
                        idx_c.append(i)
                else:
                    if revs[i]:
                        idx_t_r.append(i)
                    else:
                        idx_t.append(i)
            rev_count += len(idx_t_r)
            if len(idx_t) > 0:
                seq_l = seq_lens[idx_t]
                sent = sents[idx_t]
                pos = poss[idx_t]
                ft = fts[idx_t]
                l_start = lidx_start[idx_t]
                l_end = lidx_end[idx_t]
                r_start = ridx_start[idx_t]
                r_end = ridx_end[idx_t]
                out, prob = model(seq_l, (sent, pos, ft), l_start, l_end, 
                                  r_start, r_end, flip=False, causal=False)
                label = labels[idx_t]
                gt_labels = torch.cat([gt_labels, label.cpu()], dim=0)
                loss = criterion(out, label)
                losses_t = torch.cat([losses_t, loss.cpu()], dim=0)
                probs = torch.cat([probs, prob.cpu()], dim=0)
            if len(idx_t_r) > 0:
                seq_l = seq_lens[idx_t_r]
                sent = sents[idx_t_r]
                pos = poss[idx_t_r]
                ft = fts[idx_t_r]
                l_start = lidx_start[idx_t_r]
                l_end = lidx_end[idx_t_r]
                r_start = ridx_start[idx_t_r]
                r_end = ridx_end[idx_t_r]
                out, prob = model(seq_l, (sent, pos, ft), l_start, l_end, 
                                  r_start, r_end, flip=True, causal=False)
                label = labels[idx_t_r]
                gt_labels_rev = torch.cat([gt_labels_rev, label.cpu()], dim=0)
                loss = criterion(out, label)
                losses_t = torch.cat([losses_t, loss.cpu()], dim=0)
                probs_rev = torch.cat([probs_rev, prob.cpu()], dim=0)
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
                loss = criterion(out, label)
                losses_c = torch.cat([losses_c, loss.cpu()], dim=0)
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
                loss = criterion(out, label)
                losses_c = torch.cat([losses_c, loss.cpu()], dim=0)
            step += 1
        
        def get_reverse_label(idx_ori):
            label_ori = self._id_to_label[idx_ori]
            idx_reverse = self._label_to_id[rev_map[label_ori]]
            return idx_reverse

        if len(gt_labels_rev)!=0:
            assert len(gt_labels) == len(gt_labels_rev)
            if args.trainon=='bothWselect':
                max_probs_f = (probs.data.max(1)[0]).cpu().reshape(-1, 1)
                max_probs_b = (probs_rev.data.max(1)[0]).cpu().reshape(-1, 1)
                max_probs = torch.cat((max_probs_f, max_probs_b), 1)

                max_label_f = (probs.data.max(1)[1]).cpu().reshape(-1, 1)
                max_label_b = (probs_rev.data.max(1)[1]).cpu().reshape(-1, 1)
                max_label = torch.cat((max_label_f, max_label_b), 1)
            
                # mask decides to take forward or backward
                mask = list(max_probs.max(1)[1].view(-1).numpy())
                pred_labels_forward = torch.LongTensor([max_label[i, j] if j == 0
                                                        else get_reverse_label(max_label.data.numpy()[i, j])
                                                        for i,j in enumerate(mask)])
                pred_labels_backward = torch.LongTensor([max_label[i, j] if j == 1
                                                         else get_reverse_label(max_label.data.numpy()[i, j])
                                                         for i,j in enumerate(mask)])
            else:
                pred_labels_forward = probs.data.max(1)[1].cpu().long().view(-1)
                pred_labels_backward = probs_rev.data.max(1)[1].cpu().long().view(-1)
            
            gt_labels_f = gt_labels.data.cpu().long().view(-1)
            gt_labels_b = gt_labels_rev.data.cpu().long().view(-1)
            # simple test on whether the reverse label is correct or not
            #labels_f_rev = torch.LongTensor([get_reverse_label(label.cpu().item()) for label in gt_labels_f])
            #for b,rev in zip(gt_labels_b, labels_f_rev):
            #    assert b==rev

            # choose to test on forward or backward
            if in_dev and args.devbytrain:
                final_pred_labels = torch.cat((pred_labels_forward, pred_labels_backward), dim=0)
                final_gt_labels = torch.cat((gt_labels_f, gt_labels_b), dim=0)
            else:
                if args.teston=='forward':
                    final_pred_labels = pred_labels_forward
                    final_gt_labels = gt_labels_f
                elif args.teston=='backward':
                    final_pred_labels = pred_labels_backward
                    final_gt_labels = gt_labels_b
                elif args.teston=='bothway':
                    final_pred_labels = torch.cat((pred_labels_forward, pred_labels_backward), dim=0)
                    final_gt_labels = torch.cat((gt_labels_f, gt_labels_b), dim=0)
        else:
            final_pred_labels = probs.data.max(1)[1].cpu().long().view(-1)
            final_gt_labels = gt_labels.data.cpu().long().view(-1)

           
        if args.joint and (len(losses_c)>0):
            print("Evaluation causal loss: %.4f; accuracy: %.4f" % (torch.mean(losses_c).item(), float(correct)/float(len(losses_c))))
        return final_pred_labels, torch.mean(losses_t).item(), final_gt_labels

    def _train(self, train_data, eval_data, emb, pos_emb, args, in_cv=False):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        model = BiLSTM(emb, pos_emb, args)
        if args.cuda and torch.cuda.is_available():
            model = togpu(model)
        if args.sparse_emb and args.train_pos_emb:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-7)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        best_eval_f1 = 0.0 
        best_epoch = 0
        if args.load_model == True:
            checkpoint = torch.load(args.ilp_dir + args.load_model_file)
            model.load_state_dict(checkpoint['state_dict'])
            best_eval_f1 = checkpoint['f1']
            print("Local best eval f1 is: %s" % best_eval_f1)

        print('total train_data steps', len(train_data))
        early_stop_counter = 0
        for epoch in range(args.epochs):
            if not in_cv:
                print('Train %s epoch...' %(epoch+1))
            model.train()
            correct = 0.
            step = 1
            total_loss_s = 0.
            total_loss_c = 0.
            count_s = 0.
            count_c = 0.
            start_time = time.time()
            for data in train_data:
                seq_lens,data_id,_,labels,sents,poss,fts,revs,lidx_start,lidx_end,ridx_start,ridx_end,_ = togpu_data(data)
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
                model.zero_grad()
                loss_s = 0
                loss_c = 0
                # supervised learning
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
                    l = criterion(out, label)
                    loss_s += l
                    total_loss_s += l.item()
                    count_s += len(idx_l)
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
                    l = criterion(out, label)
                    loss_s += l
                    total_loss_s += l.item()
                    count_s += len(idx_l_r)
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
                    l = criterion(out, label)
                    loss_c += l
                    total_loss_c += l.item()
                    count_c += len(idx_c)
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
                    l = criterion(out, label)
                    loss_c += l
                    total_loss_c += l.item()
                    count_c += len(idx_c_r)

                loss = loss_s + loss_c
                loss.backward()                                                           
                optimizer.step()
                if step % 400 == 0 and (not in_cv):                                    
                    print("trained %s steps" % step)
                    print("Temporal loss is %.4f" % (total_loss_s/count_s))
                    if args.joint:
                        print("Causal loss is %.4f; accuracy is %.4f" % ((total_loss_c/count_c),(float(correct)/float(count_c)))) 
                    print("%.4f seconds elapsed" % (time.time() - start_time))
                step += 1
            # Evaluate at the end of each epoch                                 
            if len(eval_data) > 0:
                pred_labels, eval_loss, gt_labels = self.predict(model, eval_data, args, in_dev=True)
                assert gt_labels.size() == pred_labels.size() 
                pred_labels = pred_labels.cpu().tolist()
                gt_labels = gt_labels.cpu().tolist()

                eval_f1, eval_f1_category = self.weighted_f1(pred_labels, gt_labels)
                
                if eval_f1 >= best_eval_f1:
                    best_eval_f1 = eval_f1
                    # do not copy intermedite model in CV to save memory
                    if not in_cv:
                        print('Save model in %s epoch' % (epoch+1))
                        print("Best Evaluation loss: %.4f; Evaluation F1: %.4f" % (eval_loss, eval_f1))
                        self.model = copy.deepcopy(model)
                    best_epoch = epoch+1
                    early_stop_counter = 0 
                if early_stop_counter >= args.earlystop:
                    break
                else:
                    early_stop_counter += 1
        print("Final Evaluation F1: %.4f" % best_eval_f1)
        print("*"*50)
        if len(eval_data) == 0 or (args.epochs==0):
            self.model = copy.deepcopy(model)
            best_epoch = args.epochs 

        if args.save_model == True and (not in_cv):
            torch.save({
                'epoch': best_epoch,
                'args': args,
                'state_dict': self.model.state_dict(),
                'f1': best_eval_f1,
                'optimizer' : optimizer.state_dict()
            }, "%s%s.pt" % (args.ilp_dir, args.save_stamp))       

        return best_eval_f1, best_epoch
    
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
                with open('best_param/selectDev_devResult_'+str(args.data_type)+
                          '_TrainOn'+str(args.trainon)+'_TestOn'+str(args.teston)+
                          '_uf'+str(args.usefeature)+
                          '_trainpos'+str(args.train_pos_emb)+'_joint'+
                          str(args.joint)+'_devbytrain'+
                          str(args.devbytrain)+'.pickle', 'wb') as f:
                    pickle.dump(sorted(param_perf, key=lambda x: x[1], reverse=True), f, pickle.HIGHEST_PROTOCOL)
        params, f1, epoch = sorted(param_perf, key=lambda x: x[1], reverse=True)[0]
        print("*" * 50)
        print("Best F1: %s" % f1)
        print("Best Parameters Are: %s " % params)
        print("Best Epoch is: %s" % epoch)
        print("*" * 50)
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
        
        train_data = EventDataset(args.data_dir+type_dir,
                                  "train", args.glove2vocab, backward_dir, args.bert_fts)
        train_generator = get_data_loader(train_data, **params)
        dev_data = EventDataset(args.data_dir+type_dir, 
                                "dev", args.glove2vocab, backward_dir, args.bert_fts)
        dev_generator = get_data_loader(dev_data, **params)
        
        seeds = [0, 10, 20]
        accumu_f1 = 0.
        accumu_epoch = 0.
        for seed in seeds:
            exec("args.%s=%s" % ('seed', seed))
            f1, epoch = self._train(train_generator, dev_generator, emb, pos_emb, args, in_cv=True)
            accumu_f1 += f1
            accumu_epoch += epoch
        avg_f1 = accumu_f1/float(len(seeds))
        avg_epoch = accumu_epoch/float(len(seeds))

        return avg_f1, avg_epoch

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
                # multi-process over only data splits due to source limitation
                with mp.Pool(processes=args.n_splits) as pool:
                    res = pool.map(partial(self.parallel_cv, emb=emb, 
                                           pos_emb=pos_emb, args=args), all_splits)
            else:
                res = []
                for split in all_splits:
                    res.append(self.parallel_cv(split, emb=emb, pos_emb=pos_emb, args=args))
            f1s = list(zip(*res))[0]
            best_epoch = list(zip(*res))[1]
            print('avg f1 score: %s, avg epoch %s'%(np.mean(f1s), np.mean(best_epoch)))
            param_perf.append((param, np.mean(f1s), np.mean(best_epoch)))
            if args.write:
                with open('best_param/cv_devResult_'+str(args.data_type)+
                          '_TrainOn'+str(args.trainon)+'_TestOn'+str(args.teston)+
                          '_uf'+str(args.usefeature)+
                          '_trainpos'+str(args.train_pos_emb)+'_joint'+
                          str(args.joint)+'_devbytrain'+str(args.devbytrain)+
                          '.pickle', 'wb') as f:
                    pickle.dump(sorted(param_perf, key=lambda x: x[1], reverse=True), f, pickle.HIGHEST_PROTOCOL)
        params, f1, epoch = sorted(param_perf, key=lambda x: x[1], reverse=True)[0]
        print("*" * 50)
        print("Best Average F1: %s" % f1)
        print("Best Parameters Are: %s " % params)
        print("Best Epoch is: %s" % epoch)
        print("*" * 50)
        return params, epoch

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

        train_data = EventDataset(args.data_dir+'%s/fold%s/'%(type_dir, split),
                                  "train", args.glove2vocab, backward_dir, args.bert_fts)
        train_generator = get_data_loader(train_data, **params)
        dev_data = EventDataset(args.data_dir+'%s/fold%s/'%(type_dir, split), 
                                "dev", args.glove2vocab, backward_dir, args.bert_fts)
        dev_generator = get_data_loader(dev_data, **params)
        
        seeds = [0, 10, 20]
        accumu_f1 = 0.
        accumu_epoch = 0.
        for seed in seeds:
            exec("args.%s=%s" % ('seed', seed))
            f1, epoch = self._train(train_generator, dev_generator, emb, pos_emb, args, in_cv=True)
            accumu_f1 += f1
            accumu_epoch += epoch
        avg_f1 = accumu_f1/float(len(seeds))
        avg_epoch = accumu_epoch/float(len(seeds))

        return avg_f1, avg_epoch

    def weighted_f1(self, pred_labels, true_labels):
        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else (float(numr)/float(denr))

        assert len(pred_labels) == len(true_labels)
        weighted_f1_scores = {}
        if 'NONE' in self._label_to_id.keys():
            num_tests = len([x for x in true_labels if x != self._label_to_id['NONE']])
        else:
            num_tests = len([x for x in true_labels])

        #print("Total samples to eval: %s" % num_tests)
        total_true = Counter(true_labels)
        total_pred = Counter(pred_labels)
        labels = list(self._id_to_label.keys())
               
        n_correct = 0
        n_true = 0
        n_pred = 0

        # f1 score is used for tcr and matres and hence exclude vague for fair
        # comaprison with prior works
        exclude_labels = ['NONE', 'VAGUE'] if len(self._label_to_id) == 4 else ['NONE']
        #exclude_labels = ['NONE', 'VAGUE'] if len(self._label_to_id) == 4 else ['NONE', 'VAGUE']
        for label in labels:
            if self._id_to_label[label] not in exclude_labels:
                true_count = total_true.get(label, 0)
                pred_count = total_pred.get(label, 0)
                n_true += true_count
                n_pred += pred_count
                
                correct_count = len([l for l in range(len(pred_labels)) 
                                     if pred_labels[l] == true_labels[l] and pred_labels[l] == label])
                n_correct += correct_count
                precision = safe_division(correct_count, pred_count)
                recall = safe_division(correct_count, true_count)
                weighted_f1_scores[self._id_to_label[label]]={
                    'precision': precision,
                    'recall': recall,
                    'f1': safe_division(2*precision*recall, precision+recall)
                }
        precision = safe_division(n_correct, n_pred) 
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2*precision*recall, precision+recall)
        preds = [self._id_to_label[x] for x in pred_labels]
        gt = [self._id_to_label[x] for x in true_labels]
        #print(ClassificationReport('Inside weighted F1', gt, preds))
        return f1_score, weighted_f1_scores

def main_local(args):
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
    print('total train_data %s samples' %len(train_data))
    train_generator = get_data_loader(train_data, **params)
    dev_data = EventDataset(args.data_dir + type_dir, "dev", 
                            args.glove2vocab, data_dir_back, args.bert_fts)
    print('total dev_data %s samples' %len(dev_data))
    dev_generator = get_data_loader(dev_data, **params)
    
    if args.bert_fts:
        data_dir_back = args.data_dir + "all_backward_bertemb/"
    else:
        data_dir_back = args.data_dir + "all_backward/"
    test_data = EventDataset(args.data_dir + type_dir, "test", args.glove2vocab, 
                             data_dir_back, args.bert_fts)
    test_generator = get_data_loader(test_data, **params)
    
    models = [NNClassifier()]
    for model in models:
        dev_f1 = model.train_epoch(train_generator, dev_generator, args)
        evaluator = Evaluator(model)
        #evaluator.for_analysis(test_generator, args)
        score = evaluator.get_score(test_generator, args)
    return float(dev_f1), float(score)
