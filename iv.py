#coding=utf-8
#!/usr/bin/python2.7

##################################################
#update_dt: 2018-05-25
#author：梁云
#usage: 计算特征iv有效性指标,对连续特征和离散特征都适用。
##################################################

import numpy as np
import pandas as pd
import math

def iv_analysis(data,label,parts = 10):
    
    colnames = ['feature_interval',#区间
            'order_num', #订单数量
            'order_ratio', #订单占比
            'overdue_num', #逾期订单数量
            'overdue_ratio', #区间逾期订单比例
            'overdue_interval_ratio', #区间逾期订单占总逾期订单比例
            'normal_num', #正常订单数量
            'normal_ratio', #正常订单占比
            'normal_interval_ratio', #区间正常订单占总正常订单比例
            'iv_value' #iv检验值，列重复
           ]
    
    df = pd.DataFrame(data,columns = ['feature'])
    df['label'] = label
    df = df.dropna()
    df = df.sort_values('feature')
    df.index = range(len(df))

    indexmax = len(df)-1
    quantile_points = [df['feature'][int(np.ceil(float(indexmax*i)/parts))]
                      for i in range(parts+1)]
    
    cut_points = list(pd.Series(quantile_points).drop_duplicates().values)
    
    # 处理只有特征一种取值的异常情况
    if len(cut_points) == 1:
        dfiv = pd.DataFrame()
        dfiv['feature_interval'] = ['[{},{}]'.format(cut_points[0],cut_points[0])]
        dfiv['order_num'] = [len(df)]
        dfiv['order_ratio'] = [1]
        dfiv['overdue_num'] = [len(df[df['label'] >= 0.5 ])]
        dfiv['overdue_ratio'] = [float(len(df[df['label'] >= 0.5 ]))/len(df)]
        dfiv['overdue_interval_ratio'] = [1]
        dfiv['normal_num'] = [len(df[df['label'] < 0.5 ])]
        dfiv['normal_ratio'] = [len(df[df['label'] < 0.5 ]) /len(df)]
        dfiv['normal_interval_ratio'] = [1]
        dfiv['iv_value'] = [np.nan]
        return(dfiv)

        
    if len(cut_points) == 2:
        cut_points = [cut_points[0],sum(cut_points)/2.0,cut_points[1]]
    points_num = len(cut_points)

    Ldf = [0]*(points_num-1)
    for i in range(0,points_num-2):
        Ldf[i] = df.loc[(df['feature']>=cut_points[i]) \
                 & (df['feature']<cut_points[i+1]),:]    
    Ldf[points_num-2] = df.loc[(df['feature']>=cut_points[points_num-2]) \
                        & (df['feature']<=cut_points[points_num-1]),:]



    dfiv = pd.DataFrame(np.zeros((points_num -1,10)),
                        columns = colnames)

    total_overdue = len(df[df['label']>=0.5])
    total_normal = len(df[df['label']<0.5])

    for i in range(0,points_num-1):
        dfiv.loc[i,'feature_interval'] = '[{},{})'.format(cut_points[i],cut_points[i+1])
        dfiv.loc[i,'order_num'] = len(Ldf[i])
        dfiv.loc[i,'order_ratio'] = len(Ldf[i])/float(len(df))
        dfiv.loc[i,'overdue_num'] = len(Ldf[i][Ldf[i]['label'] >= 0.5 ])
        dfiv.loc[i,'overdue_ratio'] = float(dfiv.loc[i,'overdue_num'])/len(Ldf[i])
        dfiv.loc[i,'overdue_interval_ratio'] = float(dfiv.loc[i,'overdue_num'])/total_overdue if total_overdue else np.nan
        dfiv.loc[i,'normal_num'] = len(Ldf[i][Ldf[i]['label'] < 0.5 ])
        dfiv.loc[i,'normal_ratio'] = float(dfiv.loc[i,'normal_num'])/len(Ldf[i])
        dfiv.loc[i,'normal_interval_ratio'] = float(dfiv.loc[i,'normal_num'])/total_normal if total_normal else np.nan

    dfiv.loc[points_num-2,'feature_interval'] = \
         '[{},{}]'.format(cut_points[points_num-2],cut_points[points_num-1])
        
    overduer = [x if x>0 else 1e-10 for x in dfiv['overdue_interval_ratio']] # 修改异常值
    normalr = [x if x>0 else 1e-10 for x in dfiv['normal_interval_ratio']] # 修改异常值
    
    iv_value = sum([(overduer[i] - normalr[i])*math.log(float(overduer[i])/normalr[i]) 
               for i in range(len(normalr))])
    
    dfiv['iv_value'] = [iv_value] * len(dfiv) 
    
    return iv_value/100.0

    