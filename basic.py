# -*- coding:utf-8 -*-
#!/usr/bin/python2.7
###########################################################
#update_dt:2018-05-29
#author:liangyun
###########################################################

import sys,math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import feature_selection
import outliers,iv

def basic_analysis(data,label):
    dfdata = pd.DataFrame(data,columns = ['feature'])
    dfdata['label'] = label
    
    colnames = [#------覆盖率------------------------#
               'not_nan_ratio',  #非空比例，通常覆盖率coverage即是它
                'not_zero_ratio', #非零比例
                'not_outlier_ratio', #非离群值比例
                #------统计值------------------------#
                'class_num', #数据类别数目
                'value_num', #非空数据数目
                'min', #最小值
                'mean',#均值
                'med', #中位数
                'most', #众数
                'max', #最大值
                #------有效性----------------------#
                'ks(continous feature)', #ks统计量，适合连续特征
                'ks_pvalue', #ks统计量的p值
                'chi2(discrete feature)', #chi2统计量，适合离散特征
                'chi2_pvalue', #chi2统计量的p值
                't(for mean)', #均值t检验
                't_pvalue' ,#均值t检验的p值
                'z(for coverage)',#覆盖率z检验，coverage指 not_zero_ratio
                'z_pvalue', #覆盖率z检验的p值
                'iv' #iv检验值
               ]
    dfanalysis = pd.DataFrame(np.ones((1,19))*np.nan,columns = colnames)
    sample_num = len(dfdata)


    #not_nan_ratio

    dfclean = dfdata.dropna()
    not_nan_ratio = len(dfclean)/float(sample_num)

    #not_zero_ratio
    dfnonzero = dfclean.loc[dfclean['feature']!=0,:]
    not_zero_ratio = len(dfnonzero)/float(sample_num)

    #not_outlier_ratio
    dfnotoutlier = dfclean.copy()
    dfnotoutlier['feature'] = outliers.drop_outliers(
                   dfclean['feature'].values,dfclean['feature'].values)
    dfnotoutlier = dfnotoutlier.dropna()
    not_outlier_ratio = len(dfnotoutlier)/float(sample_num)

    #stats values
    class_num = len(dfclean[['feature']].drop_duplicates())
    value_num = len(dfclean)
    min_value = dfclean['feature'].min()
    mean_value = dfclean['feature'].mean()
    med = dfclean['feature'].median()
    most = stats.mode(dfclean['feature'].values).mode[0]
    max_value = dfclean['feature'].max()

    #ks test
    if class_num < 5:
        ks,ks_pvalue = np.nan,np.nan
    else:
        dfclean_overdue = dfclean[dfclean['label']==1] 
        dfclean_normal = dfclean[dfclean['label']==0]
        data_overdue = dfclean_overdue['feature'].values
        data_normal = dfclean_normal['feature'].values
        try:
            ks,ks_pvalue = list(stats.ks_2samp(data_overdue,data_normal))
        except:
            ks,ks_pvalue = np.nan,np.nan

    #chi2 test
    if class_num >= 5:
        chi2,chi2_pvalue = np.nan,np.nan
    else:
        try:
            chi2,chi2_pvalue = feature_selection.chi2(
            dfclean['feature'].values.reshape(-1, 1), dfclean['label'].values)
        except:
            chi2,chi2_pvalue = np.nan,np.nan

    #t test 
    try:
        t_value,t_pvalue = list(stats.ttest_ind(data_overdue,data_normal))
    except:
        t_value,t_pvalue = np.nan,np.nan


    #z test
    dfoverdue = dfdata[dfdata['label']==1]
    dfnormal = dfdata[dfdata['label']==0]
    dfclean_overdue = dfclean[dfclean['label']==1]
    dfclean_normal = dfclean[dfclean['label']==0]
    try:
        n1,n2 = len(dfoverdue),len(dfnormal)
        m1,m2 = len(dfclean_overdue),len(dfclean_normal)
        p1,p2,p = float(m1)/n1,float(m2)/n2,float(m1+m2)/(n1+n2)
        z_value = (p1-p2)/math.sqrt(1e-20+p*(1-p)*(1.0/n1+1.0/n2))
        z_pvalue = 2*stats.norm.cdf(-abs(z_value))
    except:
        z_value,z_pvalue = np.nan,np.nan
    
    #iv test
    dfiv = iv.iv_analysis(data,label)
    iv_value = dfiv['iv_value'][0]

    #set the values
    dfanalysis.loc[0,'not_nan_ratio'] = not_nan_ratio
    dfanalysis.loc[0,'not_zero_ratio'] = not_zero_ratio
    dfanalysis.loc[0,'not_outlier_ratio'] = not_outlier_ratio

    #
    dfanalysis.loc[0,'class_num'] = class_num
    dfanalysis.loc[0,'value_num'] = value_num
    dfanalysis.loc[0,'min'] = min_value
    dfanalysis.loc[0,'mean'] = mean_value
    dfanalysis.loc[0,'med'] = med
    dfanalysis.loc[0,'most'] = most
    dfanalysis.loc[0,'max'] = max_value

    #
    dfanalysis.loc[0,'ks(continous feature)'] = ks
    dfanalysis.loc[0,'ks_pvalue'] = ks_pvalue
    dfanalysis.loc[0,'chi2(discrete feature)'] = chi2
    dfanalysis.loc[0,'chi2_pvalue'] = chi2_pvalue
    dfanalysis.loc[0,'t(for mean)'] = t_value
    dfanalysis.loc[0,'t_pvalue'] = t_pvalue
    dfanalysis.loc[0,'z(for coverage)'] = z_value
    dfanalysis.loc[0,'z_pvalue'] = z_pvalue
    dfanalysis.loc[0,'iv'] = iv_value

    return(dfanalysis)