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

## Libraries for categorical variable handling
import collections
from scipy.stats.contingency import association


def categorical_variable_data(sig_p, global_df, local_df, cat_com_elems):
    """
    Evaluate categorical variables for statistically significant differences 
    between global_df and local_df using the Chi-square test. 
    Then measure effect sizes (Cramér's V).
    
    Args:
        sig_p (float): Significance level (e.g., 0.05).
        global_df (pd.DataFrame): DataFrame representing the 'global' population.
        local_df (pd.DataFrame): DataFrame representing the 'local' subset.
        cat_com_elems (list): List of categorical feature names common to both DataFrames.

    Returns:
        tuple: 
            int -> Number of significantly different categorical features.
            list -> Normalized effect sizes (Cramér's V) for the significant features.
    """
    effect_size_ls = []  # Holds effect size (Cramér's V) for each feature
    sig_f_ls = []        # Keeps track of features found to be significant
    num_cat_ele = len(cat_com_elems)

    for ele in cat_com_elems:
        # Get the distributions for the current categorical feature
        local_distribution = local_df[ele].tolist()
        global_distribution = global_df[ele].tolist()

        # Count the frequencies in local and global
        loc_count = dict(collections.Counter(local_distribution))
        glo_count = dict(collections.Counter(global_distribution))

        # Align keys in both local and global count dicts 
        # to ensure we can form the same-length arrays
        glo_count_ks = list(glo_count.keys())
        loc_count_ks = list(loc_count.keys())
        
        # If global categories differ from local categories, add missing categories with zero counts
        if len(glo_count_ks) != len(loc_count_ks):
            add_ele_ls = list(set(glo_count_ks) - set(loc_count_ks))
            add_ele_dic = dict.fromkeys(add_ele_ls, 0)
            loc_count = dict(list(loc_count.items()) + list(add_ele_dic.items()))
        
        # Sort dictionaries so that counts align in list form
        loc_count2 = dict(sorted(loc_count.items()))
        glo_count2 = dict(sorted(glo_count.items()))
        
        # Convert them into lists to pass into statistical tests
        loc_count3 = list(loc_count2.values())
        glo_count3 = list(glo_count2.values())

        # Perform Chi-square test
        x_result = stats.chi2_contingency([loc_count3, glo_count3])

        # If p-value < adjusted significance threshold (Bonferroni correction),
        # record feature and compute effect size using Cramér's V
        if x_result[1] < sig_p / num_cat_ele:
            sig_f_ls.append(ele)
            v_value = association([loc_count3, glo_count3], method='cramer')
            effect_size_ls.append(abs(v_value))

    # Normalize effect sizes by the max (if any significant features exist)
    if len(sig_f_ls) != 0:
        max_v = max(effect_size_ls)
        if max_v != 0:
            effect_size_ls2 = [item / max_v for item in effect_size_ls]
        else:
            effect_size_ls2 = effect_size_ls
    else:
        # If no significant features, return a list with a single zero
        effect_size_ls2 = [0]

    return len(sig_f_ls), effect_size_ls2


## Calculate Cohen's d for continuous variables
from math import sqrt

def cohend(d1, d2):
    """
    Compute Cohen's d (effect size) for two independent samples.

    Args:
        d1 (list or array): Sample 1.
        d2 (list or array): Sample 2.

    Returns:
        float: Cohen's d value.
    """
    n1, n2 = len(d1), len(d2)
    var1, var2 = np.var(d1), np.var(d2)
    
    # Pooled standard deviation
    s = sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Mean difference divided by pooled std
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / s


def t_test_d_value_list(sig_p, global_df, local_df, common_elements):
    """
    Perform independent t-tests for each continuous feature between 
    global_df and local_df. Record and normalize the Cohen's d values for 
    any features that are significant.

    Args:
        sig_p (float): Significance level (e.g., 0.05).
        global_df (pd.DataFrame): Global DataFrame with continuous features.
        local_df (pd.DataFrame): Local DataFrame with continuous features.
        common_elements (list): List of continuous feature names.

    Returns:
        tuple: 
            int -> Number of significantly different continuous features.
            list -> Normalized Cohen's d values for those features.
    """
    all_d_ls = []  # Will store Cohen's d values for significant features
    ls_01 = []     # Track names of significant features
    num_ele = len(common_elements)

    for ele in common_elements:
        # Get the distributions for the current feature
        local_distribution = local_df[ele].tolist()
        global_distribution = global_df[ele].tolist()

        # Perform two-sample t-test
        t_result = stats.ttest_ind(global_distribution, local_distribution)

        # Check p-value against Bonferroni-corrected threshold
        if t_result[1] < sig_p / num_ele:
            ls_01.append(ele)
            # Calculate absolute Cohen's d if significant
            d_result = abs(cohend(global_distribution, local_distribution))
            all_d_ls.append(d_result)

    # Normalize effect sizes for significant features 
    if len(ls_01) != 0:
        max_d = max(all_d_ls)
        if max_d != 0:
            all_d_ls2 = [item / max_d for item in all_d_ls]
        else:
            all_d_ls2 = all_d_ls
    else:
        # If no significant features, return a list with a single zero
        all_d_ls2 = [0]

    return len(ls_01), all_d_ls2


def non_iid_degree(sig_p, global_df, local_df, cont_com_elems, cat_com_elems):
    """
    Compute the non-IID degree (a measure of how different 'local_df' is 
    from 'global_df'), by combining results from continuous and categorical 
    statistical tests.

    Steps:
      1) For continuous variables, perform t-tests; collect normalized Cohen's d.
      2) For categorical variables, perform Chi-square tests; collect normalized Cramér's V.
      3) Use the count of significant features (continuous + categorical) and 
         the sum of effect sizes to compute a final non-IID measure.

    Args:
        sig_p (float): Significance level (e.g., 0.05).
        global_df (pd.DataFrame): Global DataFrame.
        local_df (pd.DataFrame): Local DataFrame.
        cont_com_elems (list): List of continuous features common to both DataFrames.
        cat_com_elems (list): List of categorical features common to both DataFrames.

    Returns:
        float: A single scalar value representing the non-IID degree.
    """
    # 1) Continuous feature analysis (t-tests)
    cont_sig_f_num, cont_es_ls = t_test_d_value_list(
        sig_p, global_df, local_df, cont_com_elems
    )
    
    # 2) Categorical feature analysis (Chi-square)
    cat_sig_f_num, cat_es_ls = categorical_variable_data(
        sig_p, global_df, local_df, cat_com_elems
    )

    # Count total features and how many were significant
    all_ele_num = len(cont_com_elems) + len(cat_com_elems)
    all_sig_f_num = cont_sig_f_num + cat_sig_f_num

    # Sum of effect sizes from both continuous and categorical
    es_sum = np.sum(cat_es_ls + cont_es_ls)

    # Combine significant proportion and effect size, then take their Euclidean norm.
    # Here arr_1 = (0, 0) and arr_2 = (proportion_of_significant_features, avg_effect_size)
    arr_1 = np.array((0, 0))
    arr_2 = np.array(
        (
            all_sig_f_num / all_ele_num, 
            es_sum / all_ele_num
        )
    )
    
    # Calculate the Euclidean distance in 2D, normalized by sqrt(2) 
    # so that the max possible distance is 1.
    non_iid = np.linalg.norm(arr_1 - arr_2) / np.sqrt(2)

    return non_iid
