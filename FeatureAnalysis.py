# -*- coding:utf-8 -*-
###########################################################
#update_dt:2018-08-27
#author:liaoweichen
#contributors:liangyun,chencai
#usage:输入预处理完毕的dataframe,对全部特征进行basic_analysis、ks_analysis、psi_analysis、chi2_analysis
###########################################################

import sys,math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing
from feature_ks import get_feature_ks
from statis import getIv
import psi_ly
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from get_ks import ks
from get_ks import print_ks
import cPickle
import warnings
from sklearn.decomposition import FactorAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
import math

warnings.filterwarnings("ignore")
pd.set_option('display.height',1000)
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

'''
数据预处理类
输入：dataframe、label列名、无效列名数组
包含功能：x维度去空、浮点型与字符串型特征检验、离散特征与连续特征检验、feature encoding、one hot编码
'''
class FeaturePreprocess(object):
    def __init__(self,df=pd.DataFrame(),label='label_ever30',invalid_features = []):
        invalid_cols = ['product_name', 'idcard', 'phone', 'user_name', 'phone_md5', 'idcard_md5', 'loan_dt','loan_mth', 'bank_id_md5']
        invalid_cols.extend(invalid_features)
        invalid_cols = list(set(invalid_cols))
        valid_cols = [col for col in df.columns if col not in invalid_cols]
        valid_cols = [col for col in valid_cols if col != label]
        self.__data = df
        self.__drop_columns = invalid_cols
        self.__valid_columns = valid_cols
        self.__label = label

    # 按照特征列去空，留下特征100%覆盖的数据
    def dropNaN(self):
        print '-----------drop na by features-----------'
        drop_columns = [col for col in self.__data.columns if col not in  self.__drop_columns]
        drop_columns.append(self.__label)
        df = self.__data
        print 'df.shape before dropna :',df.shape
        df = df.dropna(subset=drop_columns,how='all',axis=0).reset_index(drop=True)
        print 'df.shape after dropna :',df.shape
        return  df

    # 检查浮点型与字符串型特征
    def check_feature_dtype(self):
        print '-----------check feature dtype-----------'
        df = self.__data
        float_features = []
        string_features = []
        valid_cols = self.__valid_columns
        for col in valid_cols:
            try:
                df[col] = df[col].astype('float')
                float_features.append(col)
            except:
                string_features.append(col)
        return float_features,string_features

    # 检查离散特征与连续特征
    # 这里的逻辑是特征不同取值数>10(不包括空)认为是连续特征，否则是离散特征
    def check_feature_is_continuous(self):
        print '-----------check feature is coutinous-----------'
        df = self.__data
        continous_features = []
        discrete_features = []
        for col in self.__valid_columns:
            df_temp = df[[col]].dropna(subset=[col],how='any',axis=0).reset_index(drop=True)
            class_num = len(df_temp.drop_duplicates())
            if class_num > 10:
                continous_features.append(col)
            else:
                discrete_features.append(col)
        return continous_features,discrete_features

    # 有大小排序顺序的字符串型特征，做数值型映射
    # 需要提供特征值映射字典mapping_dict，key是原值，value是希望映射成的值
    def feature_encoding(self,col,mapping_dict):
        def _maping(item):
            try:return mapping_dict[item]
            except:return np.nan
        df = self.__data
        df[col] = [_maping(item) for item in df[col]]
        return df

    # 对没有大小排序顺序的字符串型特征，做one_hot编码
    # columns是需要做one_hot编码的列的数组，prefix是对应列衍生出的特征的前缀名的数组
    def one_hot_encoding(self,columns,prefix):
        if len(columns)!=len(prefix):
            raise Exception('length of columns and length of prefix not equal')
        df = self.__data
        df = pd.get_dummies(df, columns=columns, prefix=prefix)
        return df

'''
特征指标分析类
输入：dataframe,label列的名字,要求数据数据均为非string型,可以直接分析及处理
包含功能：算ks、IV、覆盖率、psi、共线性
'''
class FeatureAnalysis(object):  #输入：dataframe,label列的名字,要求数据数据均为非string型,可以直接分析及处理

    def __init__(self,df=pd.DataFrame(),label='label_ever30',invalid_features = []):
        invalid_cols = ['product_name', 'idcard', 'phone', 'user_name', 'phone_md5','idcard_md5', 'loan_dt', 'loan_mth', 'bank_id_md5']
        invalid_cols.extend(invalid_features)
        invalid_cols = list(set(invalid_cols))
        drop_cols = [col for col in list(df) if col in invalid_cols]
        df = df.drop(drop_cols,axis=1)
        self.__data = df
        self.__drop_columns = invalid_cols
        self.__label = label

    # 计算特征的覆盖率
    # wf_path为空则将覆盖率降序print出来，不为空就写到制定文件里
    def cal_coverage(self,wf_path=''):
        print '-----------calculate feature coverage-----------'
        total_length = self.__data.shape[0]
        dict_coverage = {}
        for col in list(self.__data):
            coverage = self.__data.dropna(subset=[col], how='any', axis=0).shape[0] / float(total_length)
            dict_coverage[col] = coverage
        dict_coverage = sorted(dict_coverage.items(), key=lambda x: x[1], reverse=True)
        if wf_path != '':
            wf = open(wf_path, 'w')
            for item in dict_coverage:
                wf.write(str(item[0])+'\t'+str(item[1]) + '\n')
            wf.close()
        else:
            for item in dict_coverage:
                print item[0],item[1]

    # 计算特征的ks
    # how='all'表示对所有特征计算ks，how='parts表示只对部分特征算ks，计算的特征放在features数据里
    # if_fillna=1，则对特征填充空值后算ks，填充值为fill_value,默认-1，否则对特征去空后算ks，即只算命中样本的ks
    def cal_ks_iv(self,how='all',features = [],if_fillna=1,fill_value=-1):
        if how!='all' and how!= 'parts':
            raise Exception("how can olny be 'all' or 'parts'")
        if if_fillna!=1 and if_fillna!= 0:
            raise Exception("if_fillna can olny be 1 or 0")
        print '-----------calculate feature ks iv coverage-----------'
        df = self.__data
        drop_features = self.__drop_columns
        drop_features.append(self.__label)
        drop_features = list(set(drop_features))
        if how=='all':
            valid_features = [f for f in df.columns if f not in drop_features]
        else:
            valid_features = [f for f in features if f not in drop_features]
        print 'len(valid_features):', len(valid_features)
        result_table = []
        if if_fillna==1:
            df_temp = df.fillna(fill_value).reset_index(drop=True)
            for col in valid_features:
                try:
                    coverage_rate = df[[col]].dropna(how='any', axis=0).shape[0] / float(len(df))
                    df_temp[col] = df_temp[col].astype('float')
                    KS_table = get_feature_ks(df_temp, col, self.__label)
                    KS_table = KS_table.sort_index()
                    iv_value = getIv(df_temp[self.__label],df[col])
                    print 'pass ', col, ' KS:', max(KS_table['KS_']), ' IV:',iv_value ,' coverage_rate:', coverage_rate
                    print KS_table
                    result_table.append([col, max(KS_table['KS_']), iv_value, coverage_rate])
                except:
                    continue
        else:
            for col in valid_features:
                try:
                    coverage_rate = df[[col]].dropna(how='any', axis=0).shape[0] / float(len(df))
                    df_temp = df.dropna(subset=[col], how='any', axis=0).reset_index(drop=True)
                    df_temp[col] = df_temp[col].astype('float')
                    KS_table = get_feature_ks(df_temp, col, self.__label)
                    KS_table = KS_table.sort_index()
                    iv_value = getIv(self.__label, df[col])
                    print 'pass ', col, ' KS:', max(KS_table['KS_']), ' IV:', iv_value, ' coverage_rate:', coverage_rate
                    print KS_table
                    result_table.append([col, max(KS_table['KS_']), iv_value, coverage_rate])
                except:
                    continue
        result_table = pd.DataFrame(data=result_table,columns=['feature','KS','IV','coverage'])
        result_table['KS'] = result_table['KS'].astype('float')
        result_table['coverage'] = result_table['coverage'].astype('float')
        result_table = result_table.sort_values(by='KS', ascending = False).reset_index(drop=True)
        return result_table

    # 计算两个样本的psi
    # prefix是两个样本的名称前缀
    def cal_psi_ly(self,df1=pd.DataFrame(),df2=pd.DataFrame(),prefix=['','']):
        if len(prefix)!=2:
            raise Exception('the length of prefix must be 2')
        prefix = map(str,prefix)
        psi_name = '_'.join(['psi', prefix[0], prefix[1]])
        print '-----------calculate feature psi-----------'
        result_table = []
        for col in df1.columns:
            if col not in self.__drop_columns and col!=self.__label:
                psi_value = psi_ly.psi_analysis(df1[col].values,df2[col].values)
                try:
                    result_table.append([col,float('%.4f' % psi_value)])
                except:
                    result_table.append([col,np.nan])
        result_table = pd.DataFrame(data=result_table,columns=['feature',psi_name])
        result_table = result_table.sort_values(by=psi_name, ascending=False).reset_index(drop=True)
        return result_table

    # 计算数值型特征的共线性
    def cal_correlation(self):
        df = self.__data.drop([self.__label],axis=1).fillna(-1)
        drop_columns = []
        for col in df.columns:
            try:
                df[col] = df[col].astype('float')
            except:
                drop_columns.append(col)
        df = df.drop(drop_columns,axis=1)
        cor_matrix = pd.DataFrame(np.corrcoef(df.as_matrix().transpose()), columns=df.columns, index=df.columns)
        print 'cor_matrix.shape:', cor_matrix.shape
        # 下面部分不是必须
        # arr = []
        # for row in range(len(cor_matrix)):
        #     for col in range(row):
        #         if cor_matrix.ix[row, col] > 0.7:
        #             arr.append('\t'.join([cor_matrix.index[row], cor_matrix.columns[col], str(cor_matrix.ix[row, col])]))
        return cor_matrix

'''
GBDT模型类
'''
GBDT_parameter = {
    'learning_rate':0.1,
    'n_estimators':100,
    'subsample':1.0,
    'max_depth':4,
    'min_samples_leaf':60,
    'min_samples_leaf ':0.07,
    'max_features':0.7
}
class ModelGBDT(object):
    def __init__(self,df_train,df_test,label='label_ever30',fillna_value=-1,invalid_features = [],parameter={'learning_rate':0.1,'n_estimators':100,'subsample':1.0,'max_depth':4,'min_samples_leaf':60,'min_samples_leaf ':0.07,'max_features':0.7}):
        try:Learning_rate = parameter['learning_rate']
        except:pass
        try:N_estimators = parameter['n_estimators']
        Subsample = parameter['subsample']
        Max_depth = parameter['max_depth']
        Min_samples_leaf = parameter['min_samples_leaf']
        for key in parameter:
            value = parameter[value]
        self.__train = df_train
        self.__test = df_test
        self.__x_train = df_train.drop([label], axis=1).fillna(fillna_value).reset_index(drop=True)
        self.__y_train = df_train[[label]].fillna(fillna_value).reset_index(drop=True)
        self.__model = GradientBoostingClassifier(learning_rate=Learning_rate, n_estimators=N_estimators, subsample=Subsample, max_depth=Max_depth, min_samples_leaf=Min_samples_leaf)

    def get_x_train(self):
        return self.__x_train

    def get_y_train(self):
        return self.__y_train

    def cross_validation_KF(self,scaler='',x=pd.DataFrame(),y=pd.DataFrame()):
        n_folds = 5
        model = self.__model
        print '---------------KF validation------------------------'
        print 'x.shape:', x.shape
        kf = KFold(n=x.shape[0], n_folds=n_folds, shuffle=True, random_state=None)
        avg_AUC,avg_KS,avg_KS_train = 0,0,0
        for train_index, test_index in kf:
            x_train, x_test = x.ix[train_index, :], x.ix[test_index, :]
            if scaler=='':
                scaler = StandardScaler().fit(x_train)
                x_train = scaler.transform(x_train)  # 标准化
                scaler = StandardScaler().fit(x_test)
                x_test = scaler.transform(x_test)  # 标准化
            y_train, y_test = np.array(y.ix[train_index, :]), np.array(y.ix[test_index, :])
            print '训练集上的逾期率:', float(sum(y_train)) / len(y_train)
            print '测试集上的逾期率:', float(sum(y_test)) / len(y_test)
            model.fit(x_train, y_train)
            y_predict = model.predict_proba(x_test)[:, 1]
            AUC = metrics.roc_auc_score(y_test, y_predict)
            avg_AUC += AUC
            KS = ks(y_test, y_predict)['ks']
            KSS = ks(y_test, y_predict)
            print_ks(KSS, "col", "label")
            avg_KS += KS
            print "test validation AUC: %f" % AUC
            print "validation KS Score: %f" % KS
            # 在训练集上测试
            y_predict = model.predict_proba(x_train)[:, 1]
            KS_train = ks(y_train, y_predict)['ks']

            avg_KS_train += KS_train
            print "training set KS Score: %f" % KS_train
            print '-------------------------------------------------------------------------------------------------------------------------------------'
        print "avg test validation AUC: %f" % (float(avg_AUC) / n_folds)
        print "avg validation KS Score: %f" % (float(avg_KS) / n_folds)
        print "avg train KS Score: %f" % (float(avg_KS_train) / n_folds)