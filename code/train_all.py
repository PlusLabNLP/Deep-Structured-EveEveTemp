import argparse
import time
import numpy as np
import pickle
from local_train import main_local
from global_inference import main_global
from sklearn.model_selection import ParameterGrid
from os import path
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    # share arguments:
    p.add_argument('-data_type', type=str, default="tbd")
    p.add_argument('-emb', type=int, default=300)
    p.add_argument('-num_causal', type=int, default=2)
    p.add_argument('-attention', type=str2bool, default=False)
    p.add_argument('-usefeature', type=str2bool, default=False)
    p.add_argument('-sparse_emb', type=str2bool, default=False)
    p.add_argument('-train_pos_emb', type=str2bool, default=False)
    p.add_argument('-trans_only', type=str2bool, default=False)
    p.add_argument('-trainon', type=str, default='bothway',
                   choices=['forward', 'bothway', 'bothWselect'])
    p.add_argument('-teston', type=str, default='bothway',
                   choices=['forward', 'bothway', 'backward'])
    p.add_argument('-bert_fts', type=str2bool, default=True)
    p.add_argument('-bert_dim', type=int, default=768)
    p.add_argument('-n_fts', type=int, default=15)
    p.add_argument('-n_splits', type=int, default=5)
    p.add_argument('-cuda', type=str2bool, default=True)
    p.add_argument('-joint', type=str2bool, default=False) # True only when dataset == 'new' (tcr)
    p.add_argument('-cv_shuffle', type=str2bool, default=False)
    p.add_argument('-devbytrain', type=str2bool, default=False)
    p.add_argument('-cv', type=str2bool, default=False)
    p.add_argument('-selectparam', type=str2bool, default=False)
    p.add_argument('-refit_all', type=str2bool, default=False)
    p.add_argument('-ilp_dir', type=str, default="../ILP/bert/")
    p.add_argument('-write', type=str2bool, default=False)
    args_local = p.parse_args()
    args_global = p.parse_args()
    if args_local.trans_only:
        args_local.ilp_dir="../ILP/bert_transonly/"
        args_global.ilp_dir="../ILP/bert_transonly/"
    # local arguments
    if args_local.data_type == "new":
        args_local.data_dir = "../output_data/tcr_output/"
        args_local.train_docs = [x.strip() for x in open("%strain_docs.txt" % args_local.data_dir, 'r')] 
    elif args_local.data_type == "matres":
        args_local.data_dir = "../output_data/matres_output/"
        args_local.train_docs = [x.strip() for x in open("%strain_docs.txt" % args_local.data_dir, 'r')]
        args_local.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args_local.data_dir, 'r')]
    elif args_local.data_type == "tbd":
        args_local.data_dir = "../output_data/tbd_output/"
        args_local.train_docs = [x.strip() for x in open("%strain_docs.txt" % args_local.data_dir, 'r')]
        args_local.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args_local.data_dir, 'r')]

    args_local.data_dir_u = "../output_data/te3sv_output/"
    tags = open("../output_data/tcr_output/pos_tags.txt")
    pos2idx = {}
    idx = 0
    for tag in tags:
        tag = tag.strip()
        pos2idx[tag] = idx
        idx += 1
    args_local.pos2idx = pos2idx
    
    args_local.emb_array = np.load(args_local.data_dir + 'all' + '/emb_reduced.npy', allow_pickle=True)
    args_local.glove2vocab = np.load(args_local.data_dir + 'all' + '/glove2vocab.npy', allow_pickle=True).item()
    args_local.epochs = 35
    args_local.earlystop = 7
    args_local.save_model = True
    args_local.load_model = False
    args_local.load_model_file = ''
    
    # global arguments
    if args_global.data_type == "new":
        args_global.data_dir = "../output_data/tcr_output/"
        args_global.train_docs = [x.strip() for x in open("%strain_docs.txt" % args_global.data_dir, 'r')] 
    elif args_global.data_type == "matres":
        args_global.data_dir = "../output_data/matres_output/"
        args_global.train_docs = [x.strip() for x in open("%strain_docs.txt" % args_global.data_dir, 'r')]
        args_global.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args_global.data_dir, 'r')]
    elif args_global.data_type == "tbd":
        args_global.data_dir = "../output_data/tbd_output/"
        args_global.train_docs = [x.strip() for x in open("%strain_docs.txt" % args_global.data_dir, 'r')]
        args_global.dev_docs = [x.strip() for x in open("%sdev_docs.txt" % args_global.data_dir, 'r')]

    args_global.data_dir_u = "../output_data/te3sv_output/"
    tags = open("../output_data/tcr_output/pos_tags.txt")
    pos2idx = {}
    idx = 0
    for tag in tags:
        tag = tag.strip()
        pos2idx[tag] = idx
        idx += 1
    args_global.pos2idx = pos2idx
    
    args_global.emb_array = np.load(args_global.data_dir + 'all' + '/emb_reduced.npy', allow_pickle=True)
    args_global.glove2vocab = np.load(args_global.data_dir + 'all' + '/glove2vocab.npy', allow_pickle=True).item()
    args_global.batch = 16
    args_global.epochs = 10
    args_global.optimizer = 'SGD'
    args_global.seed = 123
    args_global.earlystop = 4
    args_global.save_model = False
    args_global.load_model = True
    if args_local.data_type == 'matres':
        args_local.params = {
            'hid': [40],
            'num_layers': [2],
            'dropout': [0.7],
            'lr': [0.002],
            'batch': [32],
            'seed': [300]
        }
        args_global.params = {
            'lr': [0.08],
            'decay': [0.7],
            'momentum': [0.9],
            'margin': [0.1]
        }
    elif args_local.data_type == 'new':
        args_local.params = {
            'hid': [30],
            'num_layers': [1],
            'dropout': [0.5],
            'lr': [0.002],
            'batch': [16],
            'seed': [300]
        }
        args_global.params = {
            'lr': [0.08],
            'decay': [0.9],
            'momentum': [0.9],
            'margin': [0.1]
        }
    else:
        # for tbd
        args_local.params = {
            'hid': [60],
            'num_layers': [1],
            'dropout': [0.5],
            'lr': [0.002],
            'batch': [16],
            'seed': [100]
        }
        args_global.params = {
            'lr': [0.05],
            'decay': [0.7],
            'momentum': [0.9],
            'margin': [0.1]
        }
    s_time = time.time()
    for param_local in ParameterGrid(args_local.params):
        param_str = "local parameters: \n"
        for k,v in param_local.items():
            param_str += "%s=%s" %(k,v)
            param_str += ' '
        for k,v in param_local.items():
            exec('args_local.%s=%s' % (k,v))
            exec('args_global.%s=%s' % (k,v))
        exec('args_global.%s=%s' % ('seed',123))
        save_name = "local_{}_uf{}_trainpos{}_joint{}_TrainOn{}_TestOn{}_hid{}_lr{}_ly{}_dp{}_batch{}_seed{}".\
            format(args_local.data_type, args_local.usefeature,
                   args_local.train_pos_emb, args_local.joint,
                   args_local.trainon, args_local.teston, args_local.hid, args_local.lr, 
                   args_local.num_layers, args_local.dropout, args_local.batch, args_local.seed)
        if args_global.trans_only:
            pathname = "../ILP/bert_transonly/"+str(save_name)+".pt"
        else:
            pathname = "../ILP/bert/"+str(save_name)+".pt"
        if path.exists(pathname):
            exec('args_local.epochs=0')
            exec('args_local.load_model=True')
            exec('args_local.load_model_file="%s"'%(save_name+".pt"))
            exec('args_local.save_model=False')
            dev_f1_local, test_f1_local = main_local(args_local)
            exec('args_local.epochs=35')
            exec('args_local.load_model=False')
            exec('args_local.save_model=True')
        else:
            exec('args_local.save_stamp="%s"' %(save_name))
            dev_f1_local, test_f1_local = main_local(args_local)
            print('dev_local: {} \t, test_local: {}'.format(dev_f1_local, test_f1_local))
        
        # global training
        load_model_file = save_name+'.pt'
        exec('args_global.load_model_file="%s"' %(load_model_file))
        gs_time = time.time()
        for idx, param_global in enumerate(ParameterGrid(args_global.params)):
            param_str_g = param_str+'\nglobal parameters: \n'
            for k,v in param_global.items():
                param_str_g += "%s=%s" %(k,v)
                param_str_g += ' '
            print('*'*50)
            print('Train parameters: %s'%(param_str_g))
            for k,v in param_global.items():
                exec('args_global.%s=%s' % (k,v))
            dev_f1_global, test_f1_global = main_global(args_global)
            print('dev_global: {} \t, test_global: {}'.format(dev_f1_global, test_f1_global))
    print('Total time use:', time.time()-s_time)
