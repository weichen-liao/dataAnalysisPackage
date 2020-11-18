#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn import metrics
import psi
import math
def nvl(grouped,key):
    if key in grouped.keys():
        return float(grouped[key])
    return float(0)

import sys
if sys.getdefaultencoding() != 'utf-8':
    reload(sys)
    sys.setdefaultencoding('utf-8')


'''
gi 该区间好用户数量
bi 该区间坏用户数量
g 好用户数量
b 坏用户数量
return 返回该区间（属性值）的Iv值
'''
def caculateIv(gi,bi,g,b):
    gi, bi, g, b = float(gi),float(bi),float(g),float(b)
    if gi==0 or bi==0:
        return 0
    return (gi/g-bi/b) * math.log((gi/g)/(bi/b))

def printKs(df,name,filename='',mode='w'):
    
    print_str = ''
    print_str += '%s ks = %.1f%%' % (name,df['ks'].max()*100)+'\n'
    print_str += '用户数: %s' % (sum(df['positive_negative'])) +'\n'
    print_str += '逾期率: %.1f%%' % (100*sum(df['positive'])*1.0/sum(df['positive_negative'])) +'\n'
    print_str += '\t'.join(['seq', '评分区间', '订单数', '逾期数', '正常用户数', '百分比', '逾期率', '累计坏账户占比', '累计好账户占比', 'KS统计量']) +'\n'
    for i in range(df.shape[0]):
        print_str += "%d\t%s\t%d\t%d\t%d\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%" % (i+1,df['section'][i],df['positive_negative'][i],df['positive'][i],df['negative'][i],df['positive_negative_ratio'][i]*100,df['positive_ratio'][i]*100,df['fpr'][i]*100,df['tpr'][i]*100,df['ks'][i]*100) +'\n'
    if filename =='':
        print print_str
    if filename !='':
        f = open(filename,mode)
        f.write(print_str)
        f.close()
    return print_str

'''
函数说明: 该函数可同时计算ks和iv,并返回各区间的详细情况
y_true 标签值
y_pred 预测值（或者属性列，可使用离散值和连续值）
dataType 数据类型（continues表示连续，其他值表示离散值）
pos_label 正例对应的标签

return ks值，iv值，各区间详情
'''


def getKsIv(y_true,y_pred,Kpart=10,dataType='continues',pos_label=1):
    
    y_true, y_pred = np.array(y_true).reshape(-1), np.array(y_pred).reshape(-1)
    
    y_true =[1 if i==pos_label else 0 for i in y_true]
    df = pd.DataFrame({
            'y_pred':y_pred,
            'y_true':y_true,
    })
    df = df.sort_values(by='y_pred')
    df=df.reset_index(drop=True)
    dataunique =  df.y_pred.unique()
    kpart = min(len(dataunique),Kpart)
    section,positive,negative=[],[],[]  # 区间，该区间正例数量，负例数量
    if kpart<Kpart or dataType !='continues': # 如果离散性变量或连续性变量少于设定的数量（可以按照离散性变量处理）则直接按照（预测值，真实值)分组统计出各个属性值对应的正负样本数
        grouped = df.groupby(by=['y_pred','y_true'])['y_true'].count()
        for i in dataunique:
            section.append(i)
            positive.append(nvl(grouped,(i,1)))
            negative.append(nvl(grouped,(i,0)))  
    else:
        section_length = math.ceil(df.shape[0]*1.0/kpart) # 计算出区间长度
        
        section_start_pre,section_end_pre  = 0,0 
        for i in range(kpart):
            if i*section_length > df.shape[0]-1:
                break
            section_start,section_end = df.y_pred[i*section_length],df.y_pred[min(df.shape[0]-1,(i+1)*section_length)]
            
            if i!=0 and abs(section_start_pre - section_start)<1e-5 and abs(section_end_pre - section_end) < 1e-5 or abs(section_start - section_end) < 1e-5 and abs(section_start - section_end_pre) < 1e-5:  # 该区间已添加过，或者区间长度为0,则跳过
                continue
            section_start_pre,section_end_pre  = section_start,section_end
            
            # 计算出区间起始值
            if i==0:  # 如果是第一个区间，则是左右闭合区间，取出该区间的数据，统计出该区间的正负样本数
                 grouped = df[(df.y_pred>=section_start) & (df.y_pred<=section_end)].groupby(by='y_true')['y_true'].count()
                 section.append('[%.2f,%.2f]' %(section_start,section_end))
            else: # 其他情况，则是左开右闭区间，取出该区间的数据，统计出该区间的正负样本数
            
                grouped = df[(df.y_pred>section_start) & (df.y_pred<=section_end)].groupby(by='y_true')['y_true'].count()
                
                section.append('(%.2f,%.2f]' %(section_start,section_end))
            positive.append(nvl(grouped,1))
            negative.append(nvl(grouped,0))
    positive_sum =[sum(positive[:i]) for i in range(1,len(positive)+1)]  # 累计正样本数
    negative_sum =[sum(negative[:i]) for i in range(1,len(negative)+1)]  # 累计负样本数
    
    
    result = pd.DataFrame(
            {
                "section":section,  # 区间
                "positive_negative":np.array(positive)+np.array(negative), # 该区间样本数量
                "positive":positive, # 该区间正样本数量
                "negative":negative, # 该区间负样本数量
                "positive_ratio":np.array(positive)/(np.array(positive)+np.array(negative)), # 该区间负样本比率(逾期率)
                "positive_sum":positive_sum,  # 累计正样本数量
                "negative_sum":negative_sum,  # 累计负样本数量
                "fpr":np.array(positive_sum)/sum(positive), # 累计正样本数量占所有正样本的比率
                "tpr":np.array(negative_sum)/sum(negative), # 累计负样本数量占所有负样本的比率
            })
    result['positive_negative_ratio'] = [i*1.0 /df.shape[0] for i in  result['positive_negative'] ]
    result['ks'] = [abs(result['fpr'][i]-result['tpr'][i]) for i in range(len(result['fpr']))]
    g,b = sum(negative),sum(positive)
    result['iv'] = [caculateIv(result['negative'][i],result['positive'][i],g,b) for i in range(len(result['positive']))]
    
    dic = {
            'ks':result['ks'].max(),
            'iv':result['iv'].sum(),
            'detail':result
            }
    
    return dic
    


def getAuc(y_true,y_pred):
     return metrics.roc_auc_score(y_true,y_pred)

def getKs(y_true,y_pred,Kpart=10,dataType='continues',pos_label=1):
     result = getKsIv(y_true,y_pred,Kpart,dataType,pos_label)
     result.pop('iv')
     return result
 
def getIv(y_true,y_pred,Kpart=10,dataType='continues',pos_label=1):
     result = getKsIv(y_true,y_pred,Kpart,dataType,pos_label)
     result.pop('ks')
     iv_value = result['iv']
     iv_value = float('%.4f' % iv_value)
     return iv_value
 
def getPsi(var1,var2):
    return psi.psi(var1,var2)
 
def getCover(train_x,na_value=np.nan):
    total = len(list(train_x))
    return (total-list(train_x).count(na_value)) * 1.0 / total
 
def getNotNan(train_x,na_value=np.nan):
    total = len(list(train_x))
    return (total-list(train_x).count(na_value))
    

'''
import sys 

import getparameters

if __name__ == '__main__':

    paras = getparameters.getparameters()
    print paras        
    if 'help' in paras:
        print '-input 输入文件'
        print '-choice 计算类型,feature类别是计算每个特征的ks，name 是计算按产品名称计算ks，可指定计算ks时的target_col,name_month是按产品和月份计算ks,默认是feature类型'
        print '-target_col 如果不是计算特征的ks，则可以指定要计算ks的列，不指定默认取score'
        print '-out 在choice为feature时，输出ks的值和明细文件（2个文件）指定一个文件前缀，若不指定，则用文件名代替'
        sys.exit(0)
    


    df = pd.read_csv(paras['input'],sep='\t',encoding='utf-8')
    if df.shape[1]==1:
        df = pd.read_csv(paras['input'],encoding='utf-8')
    df['month'] = df.loan_dt.apply(lambda x:x[:7])

 
    choice = paras.get('choice','feature')

    if choice =='feature':
        
        out = paras.get('out',paras['input'])
        try:
                f = open(out+'_ks','w')
                f.truncate()
                f.close()
        except:
            pass  
        ks_list = ''
        for feature in df.columns:
            if df[feature].dtype == object:
                continue
            if feature in ['name','phone','idcard','loan_dt','label']:
                continue
            ks = getKs(df['label'],df[feature].fillna(-1))
            temp = feature+'\t'+str(ks['ks']*100)+'\n'
            print temp
            ks_list+=temp
            printKs(ks['detail'],feature,out+'_ks_detail','a')
        try:
            f = open(out+'_ks','w')
            f.truncate()
            f.write(ks_list)
            f.close()
        except:
            pass
    
    target_col = paras.get('target_col','score')

    if choice =='name':
        for name in df.name.unique():
            test = df[df.name==name]
            ks = getKs(test['label'],test[target_col])
            print name,ks['ks']

    if choice =='name_month':
        for name in df.name.unique():

            months = df.month.unique()
            months.sort()
            for month in months:
                test = df[(df.name==name) & (df.month==month)]
                if test.shape[0]==0:
                    continue
                ks = getKs(test['label'],test[target_col])
                print name,month,ks['ks']
'''





    
