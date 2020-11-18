# -*- coding:utf-8 -*-

# -*- coding:utf-8 -*-
import os
import sys
import copy
import math
import numpy as np
import pandas as pd


def get_woe_iv_ks(df, bin_num=10):
    if df.shape[0] == 0:
        return [], [], -1, -1, []

    df = copy.deepcopy(df)
    f_uniq = copy.deepcopy(df['f'].drop_duplicates().get_values())
    f_uniq.sort()

    if len(f_uniq) <= bin_num:
        df['group'] = df['f']
        bin_value = list(f_uniq)
        bin_value.sort()
    else:
        f_series = sorted(df['f'].get_values())
        f_cnt = len(f_series)
        bin_ratio = np.linspace(1.0 / bin_num, 1, bin_num)
        bin_value = list(set([f_series[int(ratio * f_cnt) - 1] for ratio in bin_ratio]))
        bin_value.sort()
        if f_series[0] < bin_value[0]:
            bin_value.insert(0, f_series[0])
        df['group'] = pd.cut(df['f'], bins=bin_value, precision=8, include_lowest=True)

    del df['f']
    group_info = df.groupby('group')
    group_sum_info = group_info.sum()
    if group_sum_info.shape[0] == 0:
        return [], [], -1, -1, []
    cumsum_info = group_sum_info.cumsum()

    group_sum_info['total'] = group_sum_info['good'] + group_sum_info['bad']
    total_good = sum(group_sum_info['good'])
    total_bad = sum(group_sum_info['bad'])
    total = total_good + total_bad
    group_sum_info['sample_ratio'] = group_sum_info['total'] / total
    group_sum_info['bad_ratio'] = group_sum_info['bad'] / total_bad
    group_sum_info['good_ratio'] = group_sum_info['good'] / total_good
    group_sum_info['woe'] = map(lambda x: 0 if x == 0 else math.log(x),
                                group_sum_info['bad_ratio'] / group_sum_info['good_ratio'])
    group_sum_info['iv'] = (group_sum_info['bad_ratio'] - group_sum_info['good_ratio']) * group_sum_info['woe']
    group_sum_info['cur_bad_ratio'] = group_sum_info['bad'] / group_sum_info['total']

    group_sum_info['cum_good'] = cumsum_info['good']
    group_sum_info['cum_bad'] = cumsum_info['bad']
    group_sum_info['cum_good_ratio'] = group_sum_info['cum_good'] / total_good
    group_sum_info['cum_bad_ratio'] = group_sum_info['cum_bad'] / total_bad
    group_sum_info['ks'] = abs(group_sum_info['cum_good_ratio'] - group_sum_info['cum_bad_ratio'])

    ks = max(group_sum_info['ks'])
    iv = sum(group_sum_info['iv'])

    result = []
    result.append(
        u'range\ttotal_cnt\tbad_cnt\tbad_pc%\tgood_cnt\tgood_pc%\tWOE\tIV\tcur_bad_ratio\tcum_bad_ratio\tcum_good_ratio\tKS_')
    for row in range(len(group_sum_info)):
        out_list = [group_sum_info.index[row]]
        for key in ['total', 'bad', 'bad_ratio', 'good', 'good_ratio', 'woe', 'iv', 'cur_bad_ratio', 'cum_bad_ratio',
                    'cum_good_ratio', 'ks']:
            out_list.append(group_sum_info[key].iloc[row])
        result.append('\t'.join([str(item) for item in out_list]))
    return group_sum_info, result, ks, iv, bin_value


def get_feature_ks(df_in,feature_name,label_name):
    df = df_in[[feature_name, label_name]].reset_index(drop=True)
    # 计算ks,    pd_df是x+label列,x是单特征
    tmp_df = pd.DataFrame()
    tmp_df['good'] = 1 - df[label_name]
    tmp_df['bad'] = df[label_name]
    tmp_df['f'] = df[feature_name]
    group_info, result, ks, iv, bin_value = get_woe_iv_ks(df=tmp_df, bin_num=10)

    array = []
    for line in result:
        line = line.split('\n')[0]
        line = line.split('\t')
        array.append(line)
        # print line,len(line)

    columns = [u'range', u'total_cnt', u'bad_cnt', u'bad_pc%', u'good_cnt', u'good_pc%', u'WOE', u'IV',
               u'cur_bad_ratio', u'cum_bad_ratio', u'cum_good_ratio', u'KS_']

    KS_table = pd.DataFrame(array[1:], columns=columns).sort_values(by='KS_',ascending = False)
    return KS_table

if __name__ == '__main__':
    print '-------------------------------------------计算单特征的KS--------------------------------------'
