'''
script for submitting jobs to runjob.
your script to run on runjob should be input several paramters by using argparser
This script will generate a bash file which is submitted to runjob. this bash file will run your actual work script
output file will located in runjob_outputs/xxx.output
'''
from __future__ import print_function
import os
from itertools import product

# parameters
data_type=['tbd']
trainons=['bothway']
usefeature=[True, False]
train_pos_emb=[False]
joint=[True]
folds=[1, 2, 0, 3, 4]

cnt = 0
for dt in data_type:
    for trainon in trainons:
        for uf in usefeature:
            for tp in train_pos_emb:
                for j in joint:
                    for fold in folds:
                        if dt!='new':
                            j = False
                        save_s = "bert_{}_TrainOn{}_uf{}_trainpos{}_joint{}_fold{}".format(dt, trainon, uf, tp, j, fold)
                        bash_file = 'scripts/bert/{}.sh'.format(save_s)
                        command_template='python gridsearch_bert_sep.py -bert_fts True -data_type {} -trainon {} -usefeature {} -train_pos_emb {} -joint {} -fold {} >> {}'
                        command = command_template.format(dt, trainon, uf, tp, j, fold, (bash_file+'.out'))
                        with open( bash_file, 'w' ) as OUT:
                            OUT.write('source ~/.zshrc\n')
                            OUT.write('cd ~/Code/aaai-event-event-relation/cwc_event_event/\n')
                            OUT.write(command)
                        if cnt < 20:
                            if dt=='new':
                                qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h_rt=24:00:00,m_mem_free=7.5G,gpu=1,h=\'vista05\' -pe mt 2 {}'.format(bash_file, bash_file)
                            elif dt=='tbd':
                                qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h_rt=24:00:00,m_mem_free=8G,gpu=1,h=\'vista02\' -pe mt 2 {}'.format(bash_file, bash_file)
                            elif dt=='matres':
                                qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h_rt=24:00:00,m_mem_free=5G,gpu=1,h=\'vista05\' -pe mt 2 {}'.format(bash_file, bash_file)
                        else:
                            if dt=='new':
                                qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h_rt=24:00:00,m_mem_free=7.5G,gpu=1,h=\'vista05\' -q ephemeral.q -pe mt 2 {}'.format(bash_file, bash_file)
                            elif dt=='tbd':
                                qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h_rt=24:00:00,m_mem_free=8G,gpu=1,h=\'vista02\' -q ephemeral.q -pe mt 2 {}'.format(bash_file, bash_file)
                            elif dt=='matres':
                                qsub_command = 'qsub -P other -j y -o {}.output -cwd -l h_rt=24:00:00,m_mem_free=5G,gpu=1,h=\'vista05\' -q ephemeral.q -pe mt 2 {}'.format(bash_file, bash_file)
                        os.system( qsub_command )
                        print( qsub_command )
                        print( 'Submitted' )
                        cnt += 1
