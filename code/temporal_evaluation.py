#!/usr/bin/python 

# this program evaluates systems that extract temporal information from text 
# tlink -> temporal links

#foreach f (24-a-gold-tlinks/data/ABC19980108.1830.0711.tml); do
#python evaluation-relations/code/temporal_evaluation.py $f $(echo $f | p 's/24-a-gold-tlinks/30-b-trips-relations/g')                             
#done

# DURING relations are changed to SIMULTANEOUS

import time 
import sys
import re 
import os

def get_arg (index):
    #for arg in sys.argv:
    return sys.argv[index]

global_prec_matched = 0 
global_rec_matched = 0 
global_system_total = 0 
global_gold_total = 0 

debug = 0
if len(sys.argv) > 4: 
	evaluation_method = get_arg(4).strip()
else: 
	evaluation_method = ''

import relation_to_timegraph
consider_DURING_as_SIMULTANEOUS = relation_to_timegraph.consider_DURING_as_SIMULTANEOUS


def extract_name(filename):
    parts = re.split('/', filename)
    length = len(parts)
    return parts[length-1]

def get_directory_path(path): 
    name = extract_name(path)
    dir = re.sub(name, '', path) 
    if dir == '': 
        dir = './'
    return dir 


def get_entity_val(word, line): 
    if re.search(word+'="[^"]*"', line): 
        entity = re.findall(word+'="[^"]*"', line)[0]
        entity = re.sub(word+'=', '', entity) 
        entity = re.sub('"', '', entity) 
        return entity 
    return word 
        
def change_DURING_relation(filetext): 
    newtext = '' 
    for line in filetext.split('\n'): 
        foo = '' 
        words = line.split('\t') 
        for i in range(0, len(words)): 
            if i == 3 and (words[i] == 'DURING' or words[i] == 'DURING_INV'): 
                foo += re.sub('DURING', 'SIMULTANEOUS', re.sub('DURING_INV', 'SIMULTANEOUS', words[i])) + '\t'
            else:
                foo += words[i] + '\t' 
        newtext += foo.strip() + '\n' 
    return newtext 

def reverse_relation(rel): 
    rel = re.sub('"', '', rel) 
    if rel.upper() == 'BEFORE': 
        return 'AFTER'
    if rel.upper() == 'AFTER': 
        return 'BEFORE' 
    if rel.upper() == 'IBEFORE': 
        return 'IAFTER' 
    if rel.upper() == 'IAFTER': 
        return 'IBEFORE' 
    if rel.upper() == 'DURING': 
        return 'DURING_INV' 
    if rel.upper() == 'BEGINS': 
        return 'BEGUN_BY' 
    if rel.upper() == 'BEGUN_BY': 
        return 'BEGINS'
    if rel.upper() == 'ENDS': 
        return 'ENDED_BY' 
    if rel.upper() == 'ENDED_BY': 
        return 'ENDS' 
    if rel.upper() == 'INCLUDES': 
        return 'IS_INCLUDED' 
    if rel.upper() == 'IS_INCLUDED': 
        return 'INCLUDES' 
    return rel.upper() 

def get_triples(tlink_file): 
    tlinks = tlink_file # open(tlink_file).read() # tlink_file # 
    relations = '' 
    for line in tlinks.split('\n'): 
        if line.strip() == '': 
            continue 
        if debug >= 4: 
            print('sending_triples', line)
        words = line.split('\t') 
        relations += words[0]+'\t'+words[1]+'\t'+words[2]+'\t'+words[3]+'\n'
        if debug >= 4: 
            print('received_triples', words[0]+'\t'+words[1]+'\t'+words[2]+'\t'+words[3]+'\n')
        if words[1] != words[2]: 
            relations += words[0]+'\t'+words[2]+'\t'+words[1]+'\t'+reverse_relation(words[3]) +'\n'        
            if debug >= 4: 
                print('received_triples', words[0]+'\t'+words[2]+'\t'+words[1]+'\t'+reverse_relation(words[3]) +'\n')
    return relations 
        
def get_timegraphs(gold, system): 
    gold_text = gold # open(gold).read() # gold #
    system_text = system # open(system).read() # system # 

    tg_gold = relation_to_timegraph.Timegraph() 
    tg_gold = relation_to_timegraph.create_timegraph_from_weight_sorted_relations(gold_text, tg_gold) 
    tg_gold.final_relations = tg_gold.final_relations + tg_gold.violated_relations
    tg_system = relation_to_timegraph.Timegraph() 
    tg_system = relation_to_timegraph.create_timegraph_from_weight_sorted_relations(system_text, tg_system) 
    tg_system.final_relations = tg_system.final_relations + tg_system.violated_relations
    return tg_gold, tg_system 

def get_x_y_rel(tlinks): 
    words = tlinks.split('\t')
    x = words[1]
    y = words[2]
    rel = words[3]
    return x, y, rel 

def get_entity_rel(tlink): 
    words = tlink.split('\t') 
    if len(words) == 3: 
        return words[0]+'\t'+words[1]+'\t'+words[2] 
    return words[1]+'\t'+words[2]+'\t'+words[3] 

def total_relation_matched(A_tlinks, B_tlinks, B_relations, B_tg): 
    count = 0 
    for tlink in A_tlinks.split('\n'): 
        if tlink.strip() == '': 
            continue 
        if debug >= 2: 
            print(tlink)
        x, y, rel = get_x_y_rel(tlink) 

        #if rel in ['VAGUE']:
        #    continue

        foo = relation_to_timegraph.interval_rel_X_Y(x, y, B_tg, rel, 'evaluation')
        if re.search(get_entity_rel(tlink.strip()), B_relations): 
            count += 1 
            if debug >= 2: 
                print('True') 
            continue 
        if debug >= 2: 
            print(x, y, rel, foo[1])
        if re.search('true', foo[1]):
            count += 1 
    return count 
           
def total_implicit_matched(system_reduced, gold_reduced, gold_tg): 
    count = 0 
    for tlink in system_reduced.split('\n'): 
        if tlink.strip() == '': 
            continue 
        if debug >= 2: 
            print(tlink)
        if re.search(tlink, gold_reduced): 
            continue 

        x, y, rel = get_x_y_rel(tlink) 
        foo = relation_to_timegraph.interval_rel_X_Y(x, y, gold_tg, rel, 'evaluation')
        if debug >= 2: 
            print(x, y, rel, foo[1])
        if re.search('true', foo[1]):
            count += 1 
    return count 
    
def get_entities(relations): 
    included = '' 
    for each in relations.split('\n'): 
        if each.strip() == '': 
            continue 
        words = each.split('\t')
        if not re.search('#'+words[1]+'#', included):
            included += '#'+words[1]+'#\n'
        if not re.search('#'+words[2]+'#', included):
            included += '#'+words[2]+'#\n'
    return included

def get_n(relations): 
    included = get_entities(relations) 
    return (len(included.split('\n'))-1)

def get_common_n(gold_relations, system_relations): 
    gold_entities = get_entities(gold_relations) 
    system_entities = get_entities(system_relations) 
    common = '' 
    for each in gold_entities.split('\n'): 
        if each.strip() == '':
            continue 
        if re.search(each, system_entities): 
            common += each + '\n' 
    if debug >= 3: 
        print(len(gold_entities.split('\n')), len(system_entities.split('\n')), len(common.split('\n')))
        print(common.split('\n'))
        print(gold_entities.split('\n'))
    return (len(common.split('\n'))-1)

def get_ref_minus(gold_relation, system_relations): 
    system_entities = get_entities(system_relations)
    count = 0 
    for each in gold_relation.split('\n'): 
        if each.strip() == '': 
            continue 
        words = each.split('\t')
        if re.search('#'+words[1]+'#', system_entities) and re.search('#'+words[2]+'#', system_entities):
            count += 1 
    return count 

def evaluate_two_files_implicit_in_recall(arg1, arg2):
    global global_prec_matched
    global global_rec_matched
    global global_system_total
    global global_gold_total

    if debug >= 1: 
        print('\n\n Evaluate', arg1, arg2)
    gold_annotation = get_relations(arg1)
    system_annotation = get_relations(arg2) 

    tg_gold, tg_system = get_timegraphs(gold_annotation, system_annotation) 
    gold_relations = get_triples(gold_annotation) 
    system_relations = get_triples(system_annotation) 
    
    #for precision
    if debug >= 2: 
        print('\nchecking precision')
    prec_matched = total_relation_matched(tg_system.final_relations, tg_gold.final_relations, gold_relations, tg_gold) 
    # for recall 
    if debug >= 2: 
        print('\nchecking recall')
    rec_matched = total_relation_matched(tg_gold.final_relations, tg_system.final_relations, system_relations, tg_system) 
    rec_implicit_matched = total_implicit_matched(tg_system.final_relations, tg_gold.final_relations, tg_gold) 
    n = get_common_n(tg_gold.final_relations, tg_system.final_relations) 
    
##    n = get_n(tg_gold.final_relations)
    ref_plus = 0.5*n*(n-1)
##    ref_minus = len(tg_gold.final_relations.split('\n'))-1
    ref_minus = rec_matched ## get_ref_minus(tg_gold.final_relations, tg_system.final_relations) 
    w = 0.99/(1+ref_plus-ref_minus) # ref_minus #
    if debug >= 2: 
        print('n =', n)
        print('rec_implicit_matched', rec_implicit_matched)
        print('n, ref_plus, ref_minus', n , ref_plus , ref_minus)
        print('w', w)
        print('rec_matched', rec_matched)
        print('total', (len(tg_gold.final_relations.split('\n'))-1))

        print('w*rec_implicit_matched', w*rec_implicit_matched)

    if debug >= 2: 
        print('precision', prec_matched, len(tg_system.final_relations.split('\n'))-1)
    if len(tg_system.final_relations.split('\n')) <= 1: 
        precision = 0 
    else: 
        precision = prec_matched*1.0/(len(tg_system.final_relations.split('\n'))-1)

    if debug >= 2: 
        print('recall', rec_matched, len(tg_gold.final_relations.split('\n'))-1)
    if len(tg_gold.final_relations.split('\n')) <= 1: 
        recall = 0 
    else:
        recall2 = (rec_matched)*1.0/(len(tg_gold.final_relations.split('\n'))-1)
        recall = (rec_matched+w*rec_implicit_matched)*1.0/(len(tg_gold.final_relations.split('\n'))-1)
        if debug >= 2: 
            print('recall2', recall2)
            print('recall', recall)


    if debug >= 1: 
        print(precision, recall, get_fscore(precision, recall))

    global_prec_matched += prec_matched
    global_rec_matched += rec_matched+w*rec_implicit_matched
    global_system_total += len(tg_system.final_relations.split('\n'))-1 
    global_gold_total += len(tg_gold.final_relations.split('\n'))-1

    return tg_system 


def evaluate_two_files(arg1, arg2):
    global global_prec_matched
    global global_rec_matched
    global global_system_total
    global global_gold_total

    if debug >= 1: 
        print('\n\nEvaluate', arg1, arg2)
    gold_annotation = arg1#get_relations(arg1)
    system_annotation = arg2#get_relations(arg2) 

    tg_gold, tg_system = get_timegraphs(gold_annotation, system_annotation) 
    gold_relations = get_triples(gold_annotation) 
    system_relations = get_triples(system_annotation) 
    
    #for precision
    if debug >= 2: 
        print('\nchecking precision')
    prec_matched = total_relation_matched(tg_system.final_relations, tg_gold.final_relations, gold_relations, tg_gold) 
    # for recall 
    if debug >= 2: 
        print('\nchecking recall')
    rec_matched = total_relation_matched(tg_gold.final_relations, tg_system.final_relations, system_relations, tg_system) 

    if debug >= 2: 
        print('precision', prec_matched, len(tg_system.final_relations.split('\n'))-1)
    if len(tg_system.final_relations.split('\n')) <= 1: 
        precision = 0 
    else: 
        precision = prec_matched*1.0/(len(tg_system.final_relations.split('\n'))-1)

    if debug >= 2: 
        print('recall', rec_matched, len(tg_gold.final_relations.split('\n'))-1)
    if len(tg_gold.final_relations.split('\n')) <= 1: 
        recall = 0 
    else:
        recall = rec_matched*1.0/(len(tg_gold.final_relations.split('\n'))-1)

    if debug >= 1: 
        print(precision, recall)

    global_prec_matched += prec_matched
    global_rec_matched += rec_matched

    system_rels = [x for x in tg_system.final_relations.split('\n')]# if 'VAGUE' not in x]
    gold_rels = [x for x in tg_gold.final_relations.split('\n')]# if'VAGUE' not in x]
    
    global_system_total += len(system_rels) - 1
    global_gold_total += len(gold_rels) - 1

    return tg_system 

def get_fscore(p, r): 
    if p+r == 0: 
        return 0 
    return 2.0*p*r/(p+r) 


def final_score(): 
    global global_prec_matched
    global global_rec_matched
    global global_system_total
    global global_gold_total

    if global_system_total == 0: 
        precision = 0 
    else: 
        precision = global_prec_matched*1.0/global_system_total
    if global_gold_total == 0: 
        recall = 0
    else: 
        recall = global_rec_matched*1.0/global_gold_total
    
    if precision == 0 and recall == 0: 
        fscore = 0 
    else: 
        fscore = get_fscore(precision, recall) 
    print('=== Temporal Awareness Score ===')
    if evaluation_method == 'acl11': 
        print('Evaluated with ACL\'11 score, not taking the reduced graph for relations.')
    elif evaluation_method == 'implicit_in_recall': 
        print('Evaluated considering implicit relations in recall as well')
    else: 
        print(evaluation_method)
    print('Temporal Score\tF1\tP\tR')
    print('\t\t'+str(100*round(fscore, 6))+'\t'+str(100*round(precision, 6))+'\t'+str(100*round(recall, 6))+'\t')
    print('Overall Temporal Awareness Score (F1 score):', str(100*round(fscore, 6)))
    print('')

    return fscore

# take input from command line and give error messages 
# call appropriate functions to evaluate 

def input_and_evaluate(): 
    invalid = 'false' 
    if len(sys.argv) < 3: 
        invalid = 'true' 
    else: 
        arg1 = get_arg(1) 
        arg2 = get_arg(2) 
        global directory_path 
        directory_path = get_directory_path(sys.argv[0])

    # both arguments are directories 
    if invalid == 'false' and os.path.isdir(arg1) and os.path.isdir(arg2): 
        # for each files in gold folder, check the performance of that file in system folder 
        if debug >= 2: 
            print('compare files in two folders')
        evaluate_two_folders(arg1, arg2)
    elif invalid == 'false' and os.path.isfile(arg1) and os.path.isfile(arg2): 
        # compare the performance between two files 
        if debug >= 2: 
            print('compare two files')

        evaluate_two_files(arg1, arg2)
        if evaluation_method == 'acl11': 
            tg = evaluate_two_files_acl11(goldfile, systemfile)
        elif evaluation_method == 'implicit_in_recall': 
            tg = evaluate_two_files_implicit_in_recall(goldfile, systemfile)
        else: 
            tg = evaluate_two_files(arg1, arg2)
    else: 
        invalid = 'true' 
        print('INVALID INPUT FORMAT')
        print('\nto check the performance of a single file:\n\t python evaluate_entities.py gold_file_path system_file_path\n')
        print('to check the performace of all files in a gold folder:\n\t python evaluate_entities.py gold_folder_path system_folder_path ')
    
    if invalid == 'false': 
        performance = 'get' 
        #get_performance() 
        final_score()



def evaluate_all(gold_files, pred_files):
    count_relation = 0 
    count_node = 0 
    count_chains = 0 
    count_time = 0

    for fl in gold_files.keys():#[0:1]:
                
        f1 = ""
        
        for words in gold_files[fl]:
            #if "VAGUE" in line:
            #    continue
            f1 += '\t'.join([str(fl)] + [str(x) for x in words[1:]])
            f1 += '\n'

        f2 = ""
        for words in pred_files[fl]:
            #if "VAGUE" in line:
            #    continue
            f2 += '\t'.join([str(fl)] + [str(x) for x in words])
            f2 += '\n'

        tg = evaluate_two_files(f1, f2)
        performance = 'get' 
    
        #count_time += end_time-start_time
        count_relation += tg.count_relation
        count_node += tg.count_node 
        count_chains += tg.next_chain+tg.count_cross_chain

    global global_prec_matched
    global global_system_total
    global global_rec_matched
    global global_gold_total
    
    print(global_prec_matched, global_system_total, global_rec_matched, global_gold_total)
    f1 = final_score()
    
    global_prec_matched = 0
    global_system_total = 0
    global_rec_matched = 0
    global_gold_total = 0
    
    return f1
