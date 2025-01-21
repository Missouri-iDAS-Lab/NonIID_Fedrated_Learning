import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from random import sample
from scipy.spatial import distance
import xgboost as xgb

## categorical variable
import collections
from scipy.stats.contingency import association

def categorical_variable_data(
    sig_p, global_df, local_df, cat_com_elems, 
    cat_f_value_ls):
    
    effect_size_ls = []
    sig_f_ls = []
    num_cat_ele = len(cat_com_elems)
    for ele in cat_com_elems:

        local_distribution = local_df[ele].tolist()
        global_distribution = global_df[ele].tolist()
        
        local_unique_value_diff = np.setdiff1d(
            cat_f_value_ls, local_df[ele].unique())

        global_unique_value_diff = np.setdiff1d(
            cat_f_value_ls, global_df[ele].unique())        
        
        loc_count = dict(
            collections.Counter(local_distribution))
        glo_count = dict(
            collections.Counter(global_distribution))

        if len(local_unique_value_diff) == 0 and len(global_unique_value_diff) == 0:
        
            glo_count_ks = list(glo_count.keys())
            loc_count_ks = list(loc_count.keys())
            if len(glo_count_ks) != len(loc_count_ks):
                add_ele_ls = list(set(glo_count_ks) - set(loc_count_ks))
                add_ele_dic = dict.fromkeys(add_ele_ls, 0)
                loc_count = dict(list(loc_count.items()) + list(add_ele_dic.items()))        

            loc_count2 = dict(sorted(loc_count.items()))
            glo_count2 = dict(sorted(glo_count.items()))
            loc_count3 = list(loc_count2.values())
            glo_count3 = list(glo_count2.values())    

            x_result = stats.chi2_contingency(
                [loc_count3, glo_count3])   
            if x_result[1] < sig_p/num_cat_ele:
                sig_f_ls.append(ele)
                v_value = association([loc_count3, glo_count3], method='cramer')
                effect_size_ls.append(abs(v_value))

        if len(local_unique_value_diff) > 0 and len(global_unique_value_diff) > 0:
            effect_size_ls.append(1)
            
    if len(sig_f_ls) != 0:     
        max_v = max(effect_size_ls)
        if max_v != 0:
            effect_size_ls2 = [item/max_v for item in effect_size_ls]
        if max_v == 0:
            effect_size_ls2 = effect_size_ls

    if len(sig_f_ls) == 0:
        effect_size_ls2 = [0]  
        
        
    return len(sig_f_ls), effect_size_ls2



## calculate Cohen's d 
from math import sqrt

def cohend(d1, d2):
    n1, n2 = len(d1), len(d2)
    var1, var2 = np.var(d1), np.var(d2)
    s = sqrt(((n1 - 1)*var1 + (n2 - 1)*var2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2)/s



def t_test_d_value_list(sig_p, global_df, 
                        local_df, common_elements):
    all_d_ls = []
    ls_01 = []
    num_ele = len(common_elements)
    for ele in common_elements:
        local_distribution = local_df[ele].tolist()
        global_distribution = global_df[ele].tolist()
        
        t_result = stats.ttest_ind(
            global_distribution, local_distribution)
        
        if t_result[1] < sig_p/num_ele:
            ls_01.append(ele)
            d_result = abs(cohend(
                global_distribution, local_distribution))
            all_d_ls.append(d_result)
        
    if len(ls_01) != 0:     
        max_d = max(all_d_ls)
        if max_d != 0:
            all_d_ls2 = [item/max_d for item in all_d_ls]
        if max_d == 0:
            all_d_ls2 = all_d_ls
            
    if len(ls_01) == 0:
        all_d_ls2 = [0]      
            
    return len(ls_01), all_d_ls2



def non_iid_degree(sig_p, global_df, local_df, 
                   cont_com_elems, cat_com_elems, 
                   cat_f_value_ls):
    
    cont_sig_f_num, cont_es_ls = t_test_d_value_list(
        sig_p, global_df, local_df, cont_com_elems)
    
    cat_sig_f_num, cat_es_ls = categorical_variable_data(
        sig_p, global_df, local_df, cat_com_elems, 
        cat_f_value_ls)

    all_ele_num = len(cont_com_elems) + len(cat_com_elems)
    all_sig_f_num = cont_sig_f_num + cat_sig_f_num
    es_sum = np.sum(cat_es_ls+cont_es_ls) 
    
#     arr_1 = np.array((0,0))
    
    arr_2 = np.array((all_sig_f_num/(2*all_ele_num), 
                      es_sum/(2*all_ele_num)))
#     non_iid = np.linalg.norm(arr_1 - arr_2)/sqrt(2)

    return sum(arr_2)


















