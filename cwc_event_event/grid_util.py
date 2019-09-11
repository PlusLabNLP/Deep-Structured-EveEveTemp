import time
def re_duplicate(best_param_list):
    final = []
    for item in best_param_list:
        if not param_equal_prev(item,final):
            final.append(item)
    return sorted(final, key=lambda x:x[-2], reverse=True)

def param_equal_prev(item, list_of_previous):
    if len(list_of_previous)==0:
        return False
    else:
        flag = False
        for p in list_of_previous:
            flag2 = True
            for k,v in p[0].items():
                if v!=item[0][k]:
                    flag2=False
            for k,v in p[1].items():
                if v!=item[1][k]:
                    flag2=False
            if flag2:
                flag=True
                return True
        return flag
