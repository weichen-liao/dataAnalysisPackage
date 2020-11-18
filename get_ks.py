# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

KS_PART = 10

def _get_cut_pos(cut_num, vec, head_pos, tail_pos):
    mid_pos = (head_pos + tail_pos) / 2
    if vec[mid_pos] == cut_num and (mid_pos == tail_pos or vec[mid_pos + 1] > cut_num):
        return mid_pos
    elif vec[mid_pos] <= cut_num:
        return _get_cut_pos(cut_num, vec, mid_pos+1, tail_pos)
    else:
        return _get_cut_pos(cut_num, vec, head_pos, mid_pos-1)

def ks(y_true, y_prob, ks_part=KS_PART):
    #print type(y_true),type(y_prob) #<type 'numpy.ndarray'>
    #print 'y_true.shape:',y_true.shape
    #print 'y_prob.shape:',y_prob.shape
    y_true = y_true.reshape(len(y_true), )
    y_prob = y_prob.reshape(len(y_prob), )
    #print 'y_true.shape:', y_true.shape
    #print 'y_prob.shape:', y_prob.shape
    data = np.vstack((y_true, y_prob)).T
    sort_ind = np.argsort(data[:, 1])
    data = data[sort_ind]

    length = len(y_prob)
    sum_bad = sum(data[:, 0])
    sum_good = length - sum_bad

    cut_list = [0]
    order_num = []
    bad_num = []

    cut_pos_last = -1
    for i in np.arange(ks_part):
        if i == ks_part-1 or data[length*(i+1)/ks_part-1, 1] != data[length*(i+2)/ks_part-1, 1]:
            cut_list.append(data[length*(i+1)/ks_part-1, 1])
            if i != ks_part-1:
                cut_pos = _get_cut_pos(data[length*(i+1)/ks_part-1, 1], data[:, 1], length*(i+1)/ks_part-1, length*(i+2)/ks_part-2)    # find the position of the rightest cut
            else:
                cut_pos = length-1
            order_num.append(cut_pos - cut_pos_last)
            bad_num.append(sum(data[cut_pos_last+1:cut_pos+1, 0]))
            cut_pos_last = cut_pos

    order_num = np.array(order_num)
    bad_num = np.array(bad_num)

    good_num = order_num - bad_num
    order_ratio = np.array([round(x, 3) for x in order_num * 100 / float(length)])
    overdue_ratio = np.array([round(x, 3) for x in bad_num * 100 / [float(x) for x in order_num]])
    bad_ratio = np.array([round(sum(bad_num[:i+1])*100/float(sum_bad), 3) for i in range(len(bad_num))])
    good_ratio = np.array([round(sum(good_num[:i+1])*100/float(sum_good), 3) for i in range(len(good_num))])
    ks_list = abs(good_ratio - bad_ratio)
    ks = max(ks_list)

    try:
        span_list = ['[%.3f,%.3f]' % (min(data[:, 1]), round(cut_list[1], 3))]
        if len(cut_list) > 2:
            for i in range(2, len(cut_list)):
                span_list.append('(%.3f,%.3f]' % (round(cut_list[i-1], 3), round(cut_list[i], 3)))
    except:
        span_list = ['0']

    dic_ks = {
            'ks': ks,
            'span_list': span_list,
            'order_num': order_num,
            'bad_num': bad_num,
            'good_num': good_num,
            'order_ratio': order_ratio,
            'overdue_ratio': overdue_ratio,
            'bad_ratio': bad_ratio,
            'good_ratio': good_ratio,
            'ks_list': ks_list
            }

    return dic_ks

def print_ks(ks_info,col,label):
    with open("ks_save_from_%s.txt"%label,'a') as f:
        f.write('col_name %s  ks = %.1f%%' % (col,ks_info['ks'])+"\n")
        f.write('\t'.join(['seq', '评分区间', '订单数', '逾期数', '正常用户数', '百分比', '逾期率', '累计坏账户占比', '累计好账户占比', 'KS统计量'])+"\n")
        for i in range(len(ks_info['ks_list'])):
            f.write('%d\t%s\t%d\t%d\t%d\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%' % (i+1, ks_info['span_list'][i], ks_info['order_num'][i], ks_info['bad_num'][i], ks_info['good_num'][i], ks_info['order_ratio'][i], ks_info['overdue_ratio'][i], ks_info['bad_ratio'][i], ks_info['good_ratio'][i], ks_info['ks_list'][i])+"\n")
        f.close()




if __name__ == '__main__':
    df_test = pd.read_csv('../data/online_test_res', sep='\t')
    dic_ks = ks(np.array(df_test['label']), np.array(df_test['prob']))
    #print_ks(dic_ks)
