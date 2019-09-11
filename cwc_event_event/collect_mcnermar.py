from collections import Counter
import pickle

def collect_mcnermar(global_seed1, global_seed2, global_seed3, 
                     local_seed1, local_seed2, local_seed3):
    g_seed1 = pickle.load(open(global_seed1, 'rb'))
    g_seed2 = pickle.load(open(global_seed2, 'rb'))
    g_seed3 = pickle.load(open(global_seed3, 'rb'))
    l_seed1 = pickle.load(open(local_seed1, 'rb'))
    l_seed2 = pickle.load(open(local_seed2, 'rb'))
    l_seed3 = pickle.load(open(local_seed3, 'rb'))
    global_dict = {**g_seed1, **g_seed2, **g_seed3}
    local_dict = {**l_seed1, **l_seed2, **l_seed3}
    #mc_result = Counter()
    error_anal_result = dict()
    for k,g_result in global_dict.items():
        #name = 'local'+k[6:]
        #l_result = local_dict[name]
        name = k[15:]
        if name in error_anal_result.keys():
            error_anal_result[name].append(g_result)
        else:
            error_anal_result[name]=[g_result]
        '''
        if g_result:
            if l_result:
                mc_result['lTrue2gTrue']+=1
                mc_result['lFalse2gTrue']+=1
        else:
            if l_result:
                mc_result['lTrue2gFalse']+=1
            else:
                mc_result['lFalse2gFalse']+=1
        '''
    for k,l_result in local_dict.items():
        name = k[14:]
        if name in error_anal_result.keys():
            error_anal_result[name].append(l_result)
        else:
            error_anal_result[name]=[l_result]
    return error_anal_result
