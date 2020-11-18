#coding=utf-8
#!/usr/bin/python2.7

##################################################
#update_dt:2018-05-28
#author：梁云
#usage:分析离群值
##################################################

import numpy as np
import pandas as pd

def outliers_analysis(data,alpha = 1.5):
    
    df = pd.DataFrame(data)
    dfclean = df.dropna()
    class_num = len(set(dfclean.iloc[:,0]))
    value_num = len(dfclean)
    
    
    colnames = [
            'med',
            'seg_25',
            'seg_75',
            'up_limit',  #异常判定上边界
            'low_limit', #异常判定下边界
            'up_ratio',  #超上边界异常值比例
            'low_ratio'  #超下边界异常值比例
            ]
    
    dfresult = pd.DataFrame(np.array([np.nan]*7).reshape(1,-1),
                            columns = colnames)
    

    if class_num<=2 or value_num <=4: return(dfresult)

    seg_25 = dfclean.quantile(0.25)[0]
    med = dfclean.median()[0]
    seg_75 = dfclean.quantile(0.75)[0]

    up_limit = seg_75 + (seg_75 - seg_25) * alpha
    low_limit = seg_25 - (seg_75 - seg_25) * alpha
    up_ratio = len(dfclean[dfclean[0]>up_limit])/float(value_num)
    low_ratio = len(dfclean[dfclean[0]<low_limit])/float(value_num)
    
    for col in colnames:dfresult.loc[0,col] = eval(col) 
        
    return(dfresult)

def drop_outliers(X_train,X_test,alpha = 1.5):
    dfresult = outliers_analysis(X_train,alpha)
    up_limit,low_limit = dfresult['up_limit'][0],dfresult['low_limit'][0]
    
    if np.isnan(up_limit) or np.isnan(low_limit):return(X_test)
    
    f = lambda x: np.nan if x<low_limit or x>up_limit else x
    return(map(f,X_test))
