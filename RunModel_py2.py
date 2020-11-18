# -*- coding:utf-8 -*-
#########################################################
# author: liaoweichen
# update_dt: 2020-07-19
#contributors:tanghuilin,liuchenzheng
#########################################################
# 训练 & 测试 & 预测模型分
#########################################################
# parameters for class
# RunModel
# 1. df 输入数据集，包含训练集与测试集
# 2. df_train,df_test 划分好的训练集与测试集
# 3. label 提前告知label特征名
# 4. invalid_features 提前告知无效的特征列名数组
# 5. scaler_switch 归一化方法，'std'对应StandardScaler，'min_max'对应MinMaxScaler，空字符串或其他表示不进行归一化

import pandas as pd
from sklearn.model_selection import KFold
# from sklearn.cross_validation import KFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.stats import norm
from scipy import stats
# import cPickle
# from sklearn2pmml import PMMLPipeline
# from sklearn2pmml import sklearn2pmml
import warnings
import copy
import math
import random

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)

percentage_list = []
for score in range(301,850):
        percentage_list.append(round(norm.cdf(score,575,68.75)*100, 4))
np.savetxt('percentage_list.txt', percentage_list, fmt = '%.4f')

def get_src_quantiles(bad_prob_array):
    bad_prob_array = np.array(bad_prob_array)
    good_prob_array = 1 - bad_prob_array
    percentage_list = np.loadtxt('percentage_list.txt')
    src_quantiles = [0]
    src_quantiles.extend(stats.scoreatpercentile(good_prob_array,percentage_list))
    #for percent in percentage_list:
    #    src_quantiles.append(np.percentile(good_prob_array, percent))
    src_quantiles.append(1)
    np.savetxt('src_quantiles.txt', src_quantiles, fmt='%.4f')

def calibrate_new(bad_prob, src_quantiles):
    good_prob = 1 - bad_prob
    if good_prob == 1:
        score = 850
    else:
        '''
        score = 300 + binary_search(good_prob, src_quantiles)
        '''
        for i in range(0, 550):
            if good_prob >= src_quantiles[i] and good_prob < src_quantiles[i + 1]:
                score = 300 + i
                break
    return score


def read_settled_probs(file_path):
    rf = open(file_path, 'r')
    prob_list = []
    for line in rf.readlines():
        prob = line.strip()
        if prob:
            prob_list.append(float(prob))
    return prob_list

def _get_cut_pos(cut_num, vec, head_pos, tail_pos):
    mid_pos = int((head_pos + tail_pos) / 2)
    if vec[mid_pos] == cut_num and (mid_pos == tail_pos or vec[mid_pos + 1] > cut_num):
        return mid_pos
    elif vec[mid_pos] <= cut_num:
        return _get_cut_pos(cut_num, vec, mid_pos+1, tail_pos)
    else:
        return _get_cut_pos(cut_num, vec, head_pos, mid_pos-1)

def ks(y_true, y_prob, ks_part=5):
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
        if i == ks_part-1 or data[int(length*(i+1)/ks_part-1), 1] != data[int(length*(i+2)/ks_part-1), 1]:
            cut_list.append(data[int(length*(i+1)/ks_part-1), 1])
            if i != ks_part-1:
                cut_pos = _get_cut_pos(data[int(length*(i+1)/ks_part-1), 1], data[:, 1], int(length*(i+1)/ks_part-1), int(length*(i+2)/ks_part-2))    # find the position of the rightest cut
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

def print_ks(ks_dict):
    data = []
    for i in range(len(ks_dict['ks_list'])):
        data.append([i+1, ks_dict['span_list'][i], ks_dict['order_num'][i], ks_dict['bad_num'][i], ks_dict['good_num'][i], ks_dict['order_ratio'][i], ks_dict['overdue_ratio'][i], ks_dict['bad_ratio'][i], ks_dict['good_ratio'][i], ks_dict['ks_list'][i]])
    data = pd.DataFrame(data=data,columns=['seq', '评分区间', '订单数', '逾期数', '正常用户数', '百分比', '逾期率', '累计坏账户占比', '累计好账户占比', 'KS统计量'])
    return data

def is_overdue_ordered(array,max_dis_order_count,max_dis_order_gap):
    def is_overdue_ordered_down(array, max_dis_order_count, max_dis_order_gap):
        ordered = True
        dis_order_count = 0
        for i in range(len(array) - 1):
            if array[i] >= array[i + 1]:
                continue
            else:
                dis_order_count += 1
                if abs(array[i] - array[i + 1]) >= max_dis_order_gap:
                    ordered = False
        if dis_order_count >= max_dis_order_count:
            ordered = False
        return ordered,dis_order_count
    def is_overdue_ordered_up(array, max_dis_order_count, max_dis_order_gap):
        ordered = True
        dis_order_count = 0
        for i in range(len(array) - 1):
            if array[i] <= array[i + 1]:
                continue
            else:
                dis_order_count += 1
                if abs(array[i] - array[i + 1]) >= max_dis_order_gap:
                    ordered = False
        if dis_order_count >= max_dis_order_count:
            ordered = False
        return ordered,dis_order_count
    if array==[]:
        return None,False
    overdue_list = array
    overdue_list = [float(item) for item in overdue_list]
    is_order_down,dis_order_count_down = is_overdue_ordered_down(overdue_list,max_dis_order_count,max_dis_order_gap)
    is_order_up, dis_order_count_up = is_overdue_ordered_up(overdue_list, max_dis_order_count, max_dis_order_gap)
    if len(overdue_list) >= 4 and (is_order_down == True or is_order_up == True):
        #print feature,overdue_list, is_overdue_ordered_up(overdue_list,max_dis_order_count,max_dis_order_gap), is_overdue_ordered_down(overdue_list,max_dis_order_count,max_dis_order_gap)
        return True,min(dis_order_count_down,dis_order_count_up)
    else:
        #print feature, overdue_list, is_overdue_ordered_up(overdue_list), is_overdue_ordered_down(overdue_list)
        return False,min(dis_order_count_down,dis_order_count_up)

# 将pickle模型文件转化为pmml格式
# path_pickle:pickle文件的路径
# path_pmml:pmml文件的路径
# explanation: 解释字段，string类型
# def pickle_to_pmml(path_pickle,path_pmml,explanation):
#     pickle_model = cPickle.load(open(path_pickle, 'rb'))
#     pipeline = PMMLPipeline([(explanation, pickle_model)])
#     sklearn2pmml(pipeline, path_pmml, with_repr=True)

class ModelGBDT(object):
    def __init__(self,df=pd.DataFrame(),df_train=pd.DataFrame(),df_test=pd.DataFrame(),label='label',invalid_features = [], random_state=1):
        invalid_cols = ['product_name', 'idcard', 'phone', 'user_name', 'phone_md5', 'idcard_md5', 'loan_dt','loan_mth', 'bank_id_md5']
        invalid_cols.extend(invalid_features)
        invalid_cols = list(set(invalid_cols))
        valid_cols = [col for col in df.columns if col not in invalid_cols]
        valid_cols = [col for col in valid_cols if col != label]
        self.__data = df
        self.__drop_columns = invalid_cols  #不包含label
        self.__valid_columns = valid_cols   #不包含label
        self.__label = label
        self.__random_state = random_state

    # 对输入数据做CV
    # apply_features为空默认所有特征进入模型，不为空则指定入模特征
    def cross_validation_KF(self, df=pd.DataFrame(), k_fold=5, apply_features=[] ,
                            scaler_switch='',fillna_value=-1,
                            n_estimators=100,learning_rate=0.1,
                            subsample=1.0,max_depth=3,
                            min_samples_split=2,min_samples_leaf=2,
                            max_features=None):
        print ("current parameters: learning_rate %s n_estimators %s subsample %s max_depth %s min_samples_split %s max_features %s min_samples_leaf %s scaler_switch %s" %(str(learning_rate),str(n_estimators),str(subsample),str(max_depth),str(min_samples_split),str(max_features),str(min_samples_leaf),scaler_switch))

        model = GradientBoostingClassifier(learning_rate=learning_rate,
                                           n_estimators=n_estimators,
                                           subsample=subsample,
                                           max_depth=max_depth,min_samples_leaf=min_samples_leaf,
                                           min_samples_split=min_samples_split,
                                           max_features=max_features,random_state=self.__random_state)
        n_folds = k_fold
        if apply_features==[]:
            apply_features = self.__valid_columns
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        else:
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        print ('len(apply_features):',len(apply_features))
        x = df[apply_features].drop([self.__label],axis=1).fillna(fillna_value).reset_index(drop=True)
        y = df[[self.__label]].reset_index(drop=True)
        print('---------------KF validation------------------------')
        # kf = KFold(n=x.shape[0], n_folds=n_folds, shuffle=True, random_state=None)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=None)
        avg_KS_test,avg_KS_train,avg_AUC_train,avg_AUC_test,count = 0,0,0,0,1
        for train_index, test_index in kf.split(x):
            print ('---------------------------fold',count,'---------------------------------')
            count += 1
            x_train, x_test = x.ix[train_index, :], x.ix[test_index, :]
            if scaler_switch == 'std':
                scaler = StandardScaler().fit(x_train)
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)
            elif scaler_switch == 'min_max':
                scaler = MinMaxScaler().fit(x_train)
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)
            y_train, y_test = np.array(y.ix[train_index, :]), np.array(y.ix[test_index, :])
            print ('训练集样本量:',len(y_train),' 训练集上的逾期率:', float(sum(y_train)) / len(y_train))
            print ('测试集样本量:',len(y_test),' 测试集上的逾期率:', float(sum(y_test)) / len(y_test))
            model.fit(x_train, y_train)
            # 在训练集上测试
            y_predict_train = model.predict_proba(x_train)[:, 1]
            dict_train = ks(y_train, y_predict_train)
            KS_train = dict_train['ks']
            avg_KS_train += KS_train
            AUC_train = metrics.roc_auc_score(y_train, y_predict_train)
            avg_AUC_train += AUC_train
            print ("training set AUC: %f" % AUC_train," training set KS Score: %f" % KS_train)
            print (print_ks(dict_train))
            # 测试集上测试
            y_predict_test = model.predict_proba(x_test)[:, 1]
            AUC_test = metrics.roc_auc_score(y_test, y_predict_test)
            avg_AUC_test += AUC_test
            dict_test = ks(y_test, y_predict_test)
            KS_test = dict_test['ks']
            avg_KS_test += KS_test
            print ("test validation AUC: %f" % AUC_test, " test validation KS Score: %f" % KS_test)
            print (print_ks(dict_test))
        print ("avg train AUC: %f" % (float(avg_AUC_train) / n_folds),"avg train KS Score: %f" % (float(avg_KS_train) / n_folds))
        print ("avg test AUC: %f" % (float(avg_AUC_test) / n_folds),"avg test KS Score: %f" % (float(avg_KS_test) / n_folds))
        print ("current parameters: learning_rate %s n_estimators %s subsample %s max_depth %s min_samples_split %s max_features %s min_samples_leaf %s scaler_switch %s" %(str(learning_rate),str(n_estimators),str(subsample),str(max_depth),str(min_samples_split),str(max_features),str(min_samples_leaf),scaler_switch))

    # 结合CV的循环调参与测试集的循环调参
    # 输入区间，循环调参
    def parameter_engine(self,df_train=pd.DataFrame(),df_test=pd.DataFrame(),
                         apply_features=[],scaler_switch='',fillna_value=-1,
                         n_folds=5,list_learning_rate = [0.1, 0.05,0.01],
                         list_n_estimators = [250,150,50], n_jobs=3,
                         list_max_depth=[4,3],list_min_samples_leaf=[8,4,2],min_samples_split=2,
                         list_subsample=[0.9,0.8],max_features=None):
        if apply_features==[]:
            apply_features = self.__valid_columns
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        else:
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        print ('len(apply_features):',len(apply_features))
        y_train = df_train[[self.__label]].reset_index(drop=True)
        x_train = df_train[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        y_test = df_test[[self.__label]].reset_index(drop=True)
        x_test = df_test[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        print ('x_test.shape:',x_test.shape)
        ret = []
        for max_depth in list_max_depth:
            for min_samples_leaf in list_min_samples_leaf:
                for subsample in list_subsample:
                    for learning_rate in list_learning_rate:
                        for n_estimators in list_n_estimators:
                            kf = KFold(n_splits=n_folds, shuffle=True, random_state=None)
                            avg_KS_test, avg_KS_train, avg_AUC_train, avg_AUC_test, count = 0, 0, 0, 0, 1
                            for train_index, test_index in kf.split(x_train):
                                count += 1
                                x_train_kf, x_test_kf = x_train.ix[train_index, :], x_train.ix[test_index, :]
                                if scaler_switch == 'std':
                                    scaler = StandardScaler().fit(x_train_kf)
                                    x_train_kf = scaler.transform(x_train_kf)
                                    x_test_kf = scaler.transform(x_test_kf)
                                elif scaler_switch == 'min_max':
                                    scaler = MinMaxScaler().fit(x_train_kf)
                                    x_train_kf = scaler.transform(x_train_kf)
                                    x_test_kf = scaler.transform(x_test_kf)
                                y_train_kf, y_test_kf = np.array(y_train.ix[train_index, :]), np.array(y_train.ix[test_index, :])
                                model_try = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators,subsample=subsample, max_depth=max_depth,min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, random_state=self.__random_state,max_features=max_features)
                                model_try.fit(x_train_kf, y_train_kf)
                                # 在CV训练集上测试
                                y_predict_train = model_try.predict_proba(x_train_kf)[:, 1]
                                dict_train = ks(y_train_kf, y_predict_train)
                                KS_train = dict_train['ks']
                                avg_KS_train += KS_train
                                # CV验证集上测试
                                y_predict_test = model_try.predict_proba(x_test_kf)[:, 1]
                                dict_test = ks(y_test_kf, y_predict_test)
                                KS_test = dict_test['ks']
                                avg_KS_test += KS_test
                            avg_validation_ks = (float(avg_KS_test) / n_folds)
                            avg_train_ks = (float(avg_KS_train) / n_folds)
                            #参数在测试集上测试
                            if scaler_switch == 'std':
                                scaler = StandardScaler().fit(x_train)
                                x_train_scaled = scaler.transform(x_train)
                                x_test_scaled = scaler.transform(x_test)
                            elif scaler_switch == 'min_max':
                                scaler = MinMaxScaler().fit(x_train)
                                x_train_scaled = scaler.transform(x_train)
                                x_test_scaled = scaler.transform(x_test)
                            else:
                                x_train_scaled = copy.deepcopy(x_train)
                                x_test_scaled = copy.deepcopy(x_test)
                            model_try = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators,subsample=subsample, max_depth=max_depth, min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,max_features=max_features,random_state=self.__random_state)
                            model_try.fit(x_train_scaled, y_train)
                            y_predict_test = model_try.predict_proba(x_test_scaled)[:, 1]
                            dict_test = ks(np.array(y_test), y_predict_test)
                            KS_test = dict_test['ks']
                            ks_table = print_ks(dict_test)
                            #检查测试集逾期率排序性
                            order,dis_order_count = is_overdue_ordered(list(ks_table['逾期率']),max_dis_order_count=0,max_dis_order_gap=0)
                            print ('max_depth:', max_depth, 'min_samples_leaf:',min_samples_leaf,'subsample:',subsample,'learning_rate:', learning_rate,'n_estimators:', n_estimators, 'avg_train_ks:', avg_train_ks, 'avg_validation_ks:', avg_validation_ks, 'test_ks:',KS_test,'is_overdue_rate_order:',order,dis_order_count)
                            ret.append({'max_depth': max_depth, 'min_samples_leaf':min_samples_leaf,'subsample':subsample,'learning_rate': learning_rate,'n_estimators': n_estimators, 'avg_train_ks': avg_train_ks, 'avg_validation_ks': avg_validation_ks, 'test_ks':KS_test})
        ret = pd.DataFrame(ret)
        return ret
    # 用输入训练集训练模型predict测试集
    # model_save_path为空默认不保存当前模型，不为空则保存模型到指定路劲
    # model_load_path为空默即时训练模型，不为空则使用指定路劲模型
    # check_feature_importance为0默认不看feature importance，为1则查看训练集的feature importance
    def test_validation(self, df_train=pd.DataFrame(),df_test=pd.DataFrame(), apply_features=[] , model_save_path='',
                        model_load_path='', scaler_switch='',fillna_value=-1,
                        check_feature_importance=0,n_estimators=100,learning_rate=0.1,
                        subsample=1.0,max_depth=3,min_samples_split=2,min_samples_leaf=6,
                        max_features=None):
        print ('---------------test validation------------------------')
        model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample,
                                           max_depth=max_depth, min_samples_split=min_samples_split,max_features=max_features,
                                           min_samples_leaf=min_samples_leaf,random_state=self.__random_state)
        if apply_features==[]:
            apply_features = self.__valid_columns
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        else:
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        print ('len(apply_features):',len(apply_features)-1)
        y_train = df_train[[self.__label]].reset_index(drop=True)
        x_train = df_train[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        y_test = df_test[[self.__label]].reset_index(drop=True)
        x_test = df_test[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        if scaler_switch == 'std':
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        elif scaler_switch == 'min_max':
            scaler = MinMaxScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        X_train,Y_train,X_test,Y_test = np.array(x_train),np.array(y_train),np.array(x_test),np.array(y_test)
        y_train, y_test = np.array(y_train), np.array(y_test)
        print ('训练集样本量:', len(y_train), ' 训练集上的逾期率:', float(sum(y_train)) / len(y_train))
        print ('测试集样本量:', len(y_test), ' 测试集上的逾期率:', float(sum(y_test)) / len(y_test))
        if model_load_path == '':
            model.fit(X_train, Y_train)
        # else:
        #     model = cPickle.load(open(model_load_path, 'rb')) #load model path
        # if model_save_path != '':
        #     cPickle.dump(model, open(model_save_path, 'wb')) #save model into save_path
        if check_feature_importance == 1:
            print ('----------------calculating feature_importance--------------')
            feature_importance = model.feature_importances_
            df_feature_importance = pd.DataFrame()
            df_feature_importance['feature'] = x_train.columns
            df_feature_importance['importance'] = feature_importance
            df_feature_importance = df_feature_importance.sort_values(['importance'], ascending=False)
            df_feature_importance = df_feature_importance.reset_index(drop=True)
            print(df_feature_importance)
            # for i in range(len(df_feature_importance)):
            #     fea = df_feature_importance.iloc[i, 'colname']
            #     fea = eval('"%s"' % fea)
            #     print (fea, df_feature_importance.iloc[i, 'importance'])
        y_predict_train = model.predict_proba(X_train)[:, 1]
        y_predict_test = model.predict_proba(X_test)[:, 1]
        dict_train = ks(y_train, y_predict_train)
        dict_test = ks(y_test, y_predict_test)

        print ('--------train---------')
        print ("time validation AUC: %f" % metrics.roc_auc_score(Y_train, y_predict_train), "validation KS Score: %f" % dict_train['ks'])
        print (print_ks(dict_train))
        print ('---------test---------')
        print ("time validation AUC: %f" % metrics.roc_auc_score(Y_test, y_predict_test), "validation KS Score: %f" % dict_test['ks'])
        print (print_ks(dict_test))
        if check_feature_importance == 1:
            return y_predict_test, df_feature_importance

    #多组随机参数训练GBDT，依据feature_importance的平均值来给特征排序
    #loop是参数的组数
    #output_feature_num是输出的特征数量，输出排名靠最前的若干个特征
    def select_features_through_GBDT_feature_importances(self,df = pd.DataFrame(),apply_features=[], scaler_switch='',fillna_value=-1,loop=10,output_feature_num=100):
        if apply_features==[]:
            apply_features = self.__valid_columns
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        else:
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        print ('len(apply_features):',len(apply_features))
        y = df[[self.__label]].reset_index(drop=True)
        x = df[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        print ('data.shape:',x.shape)
        if scaler_switch == 'std':
            scaler = StandardScaler().fit(x)
            x = scaler.transform(x)
        elif scaler_switch == 'min_max':
            scaler = MinMaxScaler().fit(x)
            x = scaler.transform(x)
        X,Y = np.array(x),np.array(y)
        dict_feature_importances = {}   #用来存每个特征的feature排序，value是一个list
        for col in x.columns:
            dict_feature_importances[col] = []
        for i in range(loop):
            print ('now calculating loop ' + str(i) + ',total is ' + str(loop) + '............')
            model = GradientBoostingClassifier(learning_rate=float('%.2f'% random.uniform(0.01,0.2)), n_estimators=random.randint(20,200),
                                               subsample=0.8,
                                               max_depth=random.randint(2,6), min_samples_split=2,
                                               max_features=1.0)
            model.fit(X, Y)
            feature_importance = model.feature_importances_
            df_feature_importance = pd.DataFrame()
            df_feature_importance['colname'] = x.columns
            df_feature_importance['importance'] = feature_importance
            df_feature_importance = df_feature_importance.sort_values(['importance'], ascending=False)
            df_feature_importance = df_feature_importance.reset_index()
            df_feature_importance['order'] = df_feature_importance.index
            for j in range(len(df_feature_importance)):
                fea = df_feature_importance.ix[j,'colname']
                order = df_feature_importance.ix[j,'order']
                dict_feature_importances[fea].append(order)
        for key in dict_feature_importances:
            dict_feature_importances[key] = np.mean(dict_feature_importances[key])
        dict_feature_importances = sorted(dict_feature_importances.items(), key=lambda x: x[1], reverse=False)  #按order升序
        ouput_features = []
        for item in dict_feature_importances:
            #print item[0],item[1]
            ouput_features.append(item[0])
        return ouput_features[:output_feature_num]


    # 一键完成分数映射相关工序
    # 调用已有的模型，保存训练集的特征列顺序，保存模型预测训练集的概率，模型预测测试集，并根据训练集的概率分布计算测试集的分数映射
    '''
    def test_validation_model(self,df_train=pd.DataFrame(),df_test=pd.DataFrame(), apply_features=[] , model_load_path='models/', save_train_cols_path='',save_prob_path='',scaler_switch='',fillna_value=-1):
        print ('---------------test validation------------------------')
        model = cPickle.load(open(model_load_path, 'rb'))  # load model path
        if apply_features == []:
            apply_features = self.__valid_columns
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        else:
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        print 'len(apply_features):', len(apply_features)
        y_train = df_train[[self.__label]].reset_index(drop=True)
        x_train = df_train[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        y_test = df_test[[self.__label]].reset_index(drop=True)
        x_test = df_test[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        if save_train_cols_path != '':  # save trainingset columns
            wf = open(save_train_cols_path, 'w')
            for col in x_train.columns:
                wf.write(col + '\n')
            wf.close()
        if scaler_switch == 'std':
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        elif scaler_switch == 'min_max':
            scaler = MinMaxScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        X_train, Y_train, X_test, Y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        y_train, y_test = np.array(y_train), np.array(y_test)
        # 模型预测训练集得到概率，并保存起来
        y_predict = model.predict_proba(X_train)[:, 1]
        wf = open(save_prob_path, 'w')
        for item in y_predict:
            wf.write(str(item) + '\n')
        wf.close()
        #模型预测测试集
        y_predict = model.predict_proba(X_test)[:, 1]
        settled_probs = read_settled_probs(save_prob_path)
        get_src_quantiles(settled_probs)
        src_quantiles = np.loadtxt('src_quantiles.txt')
        score = [calibrate_new(item, src_quantiles) for item in y_predict]  # 映射分数
        dict_test = ks(y_test, np.array(score))
        print "time validation AUC: %f" % metrics.roc_auc_score(Y_test, y_predict), "validation KS Score: %f" % dict_test['ks']
        print print_ks(dict_test)
    '''


    #调用已有模型，根据已保存的概率值映射分数，并计算ks、分箱。该函数用于检验已经生成的模型文件表现是否与训练一致
    '''
    def model_file_test(self,df_test=pd.DataFrame(),apply_features=[],model_load_path='',prob_load_path='',fillna_value=-1,scaler_switch=''):
        model = cPickle.load(open(model_load_path, 'rb'))  # load model path
        if apply_features == []:
            apply_features = self.__valid_columns
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        else:
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        print 'len(apply_features):', len(apply_features)
        y_test = df_test[[self.__label]].reset_index(drop=True)
        x_test = df_test[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        if scaler_switch == 'std':
            scaler = StandardScaler().fit(x_test)
            x_test = scaler.transform(x_test)
        elif scaler_switch == 'min_max':
            scaler = MinMaxScaler().fit(x_test)
            x_test = scaler.transform(x_test)
        X_test, Y_test = np.array(x_test), np.array(y_test)
        y_test = np.array(y_test)
        # 模型预测测试集
        y_predict = model.predict_proba(X_test)[:, 1]
        settled_probs = read_settled_probs(prob_load_path)
        get_src_quantiles(settled_probs)
        src_quantiles = np.loadtxt('src_quantiles.txt')
        score = [calibrate_new(item, src_quantiles) for item in y_predict]  # 映射分数
        dict_test = ks(y_test, np.array(score))
        print "time validation AUC: %f" % metrics.roc_auc_score(Y_test, y_predict), "validation KS Score: %f" % \
                                                                                    dict_test['ks']
        print print_ks(dict_test)
        '''



class ModelLR(object):
    def __init__(self,df=pd.DataFrame(),df_train=pd.DataFrame(),df_test=pd.DataFrame(),label='label',invalid_features = []):
        invalid_cols = ['product_name', 'idcard', 'phone', 'user_name', 'phone_md5', 'idcard_md5', 'loan_dt','loan_mth', 'bank_id_md5']
        invalid_cols.extend(invalid_features)
        invalid_cols = list(set(invalid_cols))
        valid_cols = [col for col in df.columns if col not in invalid_cols]
        valid_cols = [col for col in valid_cols if col != label]
        self.__data = df
        self.__drop_columns = invalid_cols  #不包含label
        self.__valid_columns = valid_cols   #不包含label
        self.__label = label

    # K折交叉验证
    # penalty:正则化选择参数，参数可选值为l1和l2，分别对应l1正则化和l2正则化，默认是l2正则化。 如果想进一步要泛化能力，可选l1
    # solver:损失函数优化算法,l2可选 {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’},l1可选liblinear
    # dual:用来指明是否将原问题改成他的对偶问题，对偶问题可以理解成相反问题，比如原问题是求解最大值的线性规划，那么他的对偶问题就是转化为求解最小值的线性规划，适用于样本较小的数据集，因样本小时，计算复杂度较低。 Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
    # tol:残差收敛条件，默认是0.0001，也就是只需要收敛的时候两步只差＜0.0001 就停止，可以设置更大或更小。(逻辑回归模型的损失函数是残差平方和)
    # C:正则化强度的导数，必须是一个正浮点数，值越小，正则化强度越大，即防止过拟合的程度更大;Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    # fit_intercept:是否将截距/方差加入到决策模型中，默认为True。
    # class_weight:class_weight是很重要的一个参数，是用来调节正负样本比例的，默认是值为None，也就是正负样本的权重是一样的，你可以以dict的形式给模型传入任意你认为合适的权重比，也可以直接指定一个值“balanced”，模型会根据正负样本的绝对数量比来设定模型最后结果的权重比。eg:class_weight={0:0.9, 1:0.1}
    # random_state:随机种子的设置，默认是None,如果设置了随机种子，那么每次使用的训练集和测试集都是一样的，这样不管你运行多少次，最后的准确率都是一样的；如果没有设置，那么每次都是不同的训练集和测试集，最后得出的准确率也是不一样的。
    # max_iter:算法收敛的最大迭代次数，即求取损失函数最小值的迭代次数，默认是100，越大就越容易过拟合，越小越容易欠拟合
    # multi_class:分类方法参数选择，‘ovr’二分类和‘multinomial’多分类两个值可以选择，默认值为‘ovr’
    # verbose:英文意思是”冗余“，就是会输出一些模型运算过程中的东西（任务进程），默认是False，也就是不需要输出一些不重要的计算过程。
    def cross_validation_KF(self, df=pd.DataFrame(), k_fold=5, apply_features=[] ,scaler_switch='',fillna_value=-1,penalty='l2',solver='liblinear',dual=False,tol=0.0001,C=1.0,fit_intercept=True,class_weight=None,random_state=None,max_iter=100,multi_class='ovr',verbose=0):
        print ("current parameters: penalty %s solver %s dual %s tol %s C %s fit_intercept %s class_weight %s random_state %s max_iter %s multi_class %s scaler_switch %s" %(str(penalty),str(solver),str(dual),str(tol),str(C),str(fit_intercept),str(class_weight),str(random_state),str(max_iter),str(multi_class),scaler_switch))

        model = LogisticRegression(penalty=penalty,solver=solver,dual=dual,tol=tol,C=C,fit_intercept=fit_intercept,class_weight=class_weight,random_state=random_state,max_iter=max_iter,multi_class=multi_class,verbose=verbose)
        n_folds = k_fold
        if apply_features==[]:
            apply_features = self.__valid_columns
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        else:
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        print ('len(apply_features):',len(apply_features))
        df[apply_features].drop(columns=['Class'],axis=1)
        x = df[apply_features].drop([self.__label],axis=1).fillna(fillna_value).reset_index(drop=True)
        y = df[[self.__label]].reset_index(drop=True)
        print ('---------------KF validation------------------------')
        # kf = KFold(n=x.shape[0], n_folds=n_folds, shuffle=True, random_state=None)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=None)
        avg_KS_test,avg_KS_train,avg_AUC_train,avg_AUC_test,count = 0,0,0,0,1
        for train_index, test_index in kf.split(x):
            print ('---------------------------fold',count,'---------------------------------')
            count += 1
            x_train, x_test = x.iloc[train_index, :], x.iloc[test_index, :]
            if scaler_switch == 'std':
                scaler = StandardScaler().fit(x_train)
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)
            elif scaler_switch == 'min_max':
                scaler = MinMaxScaler().fit(x_train)
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)
            y_train, y_test = np.array(y.iloc[train_index, :]), np.array(y.iloc[test_index, :])
            print ('训练集样本量:', len(y_train), ' 训练集上的逾期率:', float(sum(y_train)) / len(y_train))
            print ('测试集样本量:', len(y_test), ' 测试集上的逾期率:', float(sum(y_test)) / len(y_test))
            model.fit(x_train, y_train)
            # 在训练集上测试
            y_predict_train = model.predict_proba(x_train)[:, 1]
            dict_train = ks(y_train, y_predict_train)
            KS_train = dict_train['ks']
            avg_KS_train += KS_train
            AUC_train = metrics.roc_auc_score(y_train, y_predict_train)
            avg_AUC_train += AUC_train
            print ("training set AUC: %f" % AUC_train," training set KS Score: %f" % KS_train)
            print (print_ks(dict_train))
            # 测试集上测试
            y_predict_test = model.predict_proba(x_test)[:, 1]
            AUC_test = metrics.roc_auc_score(y_test, y_predict_test)
            avg_AUC_test += AUC_test
            dict_test = ks(y_test, y_predict_test)
            KS_test = dict_test['ks']
            avg_KS_test += KS_test
            print ("test validation AUC: %f" % AUC_test, " test validation KS Score: %f" % KS_test)
            print (print_ks(dict_test))
        print ("avg train AUC: %f" % (float(avg_AUC_train) / n_folds),"avg train KS Score: %f" % (float(avg_KS_train) / n_folds))
        print ("avg test AUC: %f" % (float(avg_AUC_test) / n_folds),"avg test KS Score: %f" % (float(avg_KS_test) / n_folds))

        print ("current parameters: penalty %s solver %s dual %s tol %s C %s fit_intercept %s class_weight %s random_state %s max_iter %s multi_class %s scaler_switch %s" % (
        str(penalty), str(solver), str(dual), str(tol), str(C), str(fit_intercept), str(class_weight),
        str(random_state), str(max_iter), str(multi_class), scaler_switch))


    # 结合CV的循环调参与测试集的循环调参
    # 输入list_max_iter、list_C区间，循环调参
    def parameter_engine(self,df_train=pd.DataFrame(),df_test=pd.DataFrame(), apply_features=[],scaler_switch='',fillna_value=-1,n_folds=5,list_max_iter = map(int,np.linspace(100, 200, 10)),list_C = np.linspace(1, 100, 19),penalty='l2',solver='liblinear',dual=False,tol=0.0001,fit_intercept=True,class_weight=None,random_state=None,multi_class='ovr',verbose=0):
        if apply_features==[]:
            apply_features = self.__valid_columns
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        else:
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        print ('len(apply_features):',len(apply_features))
        y_train = df_train[[self.__label]].reset_index(drop=True)
        x_train = df_train[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        y_test = df_test[[self.__label]].reset_index(drop=True)
        x_test = df_test[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        print ('x_test.shape:',x_test.shape)
        for max_iter in list_max_iter:
            for C in list_C:
                # kf = KFold(n=x_train.shape[0], n_folds=n_folds, shuffle=True, random_state=None)
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=None)
                avg_KS_test, avg_KS_train, avg_AUC_train, avg_AUC_test, count = 0, 0, 0, 0, 1
                for train_index, test_index in kf.split(x_train):
                    count += 1
                    x_train_kf, x_test_kf = x_train.iloc[train_index, :], x_train.iloc[test_index, :]
                    if scaler_switch == 'std':
                        scaler = StandardScaler().fit(x_train_kf)
                        x_train_kf = scaler.transform(x_train_kf)
                        x_test_kf = scaler.transform(x_test_kf)
                    elif scaler_switch == 'min_max':
                        scaler = MinMaxScaler().fit(x_train_kf)
                        x_train_kf = scaler.transform(x_train_kf)
                        x_test_kf = scaler.transform(x_test_kf)
                    y_train_kf, y_test_kf = np.array(y_train.ix[train_index, :]), np.array(y_train.ix[test_index, :])
                    model_try = LogisticRegression(penalty=penalty,solver=solver,dual=dual,tol=tol,C=C,fit_intercept=fit_intercept,class_weight=class_weight,random_state=random_state,max_iter=max_iter,multi_class=multi_class,verbose=verbose)
                    model_try.fit(x_train_kf, y_train_kf)
                    # 在CV训练集上测试
                    y_predict_train = model_try.predict_proba(x_train_kf)[:, 1]
                    dict_train = ks(y_train_kf, y_predict_train)
                    KS_train = dict_train['ks']
                    avg_KS_train += KS_train
                    # CV验证集上测试
                    y_predict_test = model_try.predict_proba(x_test_kf)[:, 1]
                    dict_test = ks(y_test_kf, y_predict_test)
                    KS_test = dict_test['ks']
                    avg_KS_test += KS_test
                avg_validation_ks = (float(avg_KS_test) / n_folds)
                avg_train_ks = (float(avg_KS_train) / n_folds)
                #参数在测试集上测试
                if scaler_switch == 'std':
                    scaler = StandardScaler().fit(x_train)
                    x_train_scaled = scaler.transform(x_train)
                    x_test_scaled = scaler.transform(x_test)
                elif scaler_switch == 'min_max':
                    scaler = MinMaxScaler().fit(x_train)
                    x_train_scaled = scaler.transform(x_train)
                    x_test_scaled = scaler.transform(x_test)
                else:
                    x_train_scaled = copy.deepcopy(x_train)
                    x_test_scaled = copy.deepcopy(x_test)
                model_try = LogisticRegression(penalty=penalty, solver=solver, dual=dual, tol=tol, C=C,
                                                   fit_intercept=fit_intercept, class_weight=class_weight,
                                                   random_state=random_state, max_iter=max_iter,
                                                   multi_class=multi_class, verbose=verbose)
                model_try.fit(x_train_scaled, y_train)
                y_predict_test = model_try.predict_proba(x_test_scaled)[:, 1]
                dict_test = ks(np.array(y_test), y_predict_test)
                KS_test = dict_test['ks']
                ks_table = print_ks(dict_test)
                #检查测试集逾期率排序性
                order,dis_order_count = is_overdue_ordered(list(ks_table['逾期率']),max_dis_order_count=0,max_dis_order_gap=0)
                print ('max_iter:', max_iter, 'C:', C, 'avg_train_ks:', avg_train_ks, 'avg_validation_ks:', avg_validation_ks, 'test_ks:',KS_test,'is_overdue_rate_order:',order,dis_order_count)


    # 用输入训练集训练模型predict测试集
    # model_save_path为空默认不保存当前模型，不为空则保存模型到指定路劲
    # model_load_path为空默即时训练模型，不为空则使用指定路劲模型
    # check_feature_importance为0默认不看feature importance，为1则查看训练集的feature importance
    def test_validation(self, df_train=pd.DataFrame(), df_test=pd.DataFrame(), apply_features=[],
                        model_save_path='', model_load_path='', scaler_switch='', fillna_value=-1,
                        penalty='l2', solver='liblinear', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                        class_weight=None, random_state=None, max_iter=100, multi_class='ovr', verbose=0):
        print ('---------------test validation------------------------')
        model = LogisticRegression(penalty=penalty, solver=solver, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                   class_weight=class_weight, random_state=random_state, max_iter=max_iter,
                                   multi_class=multi_class, verbose=verbose)
        if apply_features == []:
            apply_features = self.__valid_columns
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        else:
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        print ('len(apply_features):', len(apply_features))
        y_train = df_train[[self.__label]].reset_index(drop=True)
        x_train = df_train[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        y_test = df_test[[self.__label]].reset_index(drop=True)
        x_test = df_test[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        if scaler_switch == 'std':
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        elif scaler_switch == 'min_max':
            scaler = MinMaxScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        X_train, Y_train, X_test, Y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        y_train, y_test = np.array(y_train), np.array(y_test)
        print ('训练集样本量:', len(y_train), ' 训练集上的逾期率:', float(sum(y_train)) / len(y_train))
        print ('测试集样本量:', len(y_test), ' 测试集上的逾期率:', float(sum(y_test)) / len(y_test))
        if model_load_path == '':
            model.fit(X_train, Y_train)
        # else:
        #     model = cPickle.load(open(model_load_path, 'rb'))  # load model path
        # if model_save_path != '':
        #     cPickle.dump(model, open(model_save_path, 'wb'))  # save model into save_path

        y_predict = model.predict_proba(X_test)[:, 1]
        dict_test = ks(y_test, y_predict)
        print ("time validation AUC: %f" % metrics.roc_auc_score(Y_test, y_predict), "validation KS Score: %f" % \
                                                                                    dict_test['ks'])
        print (print_ks(dict_test))

    # 一键完成分数映射相关工序
    # 调用已有的模型，保存训练集的特征列顺序，保存模型预测训练集的概率，模型预测测试集，并根据训练集的概率分布计算测试集的分数映射
    '''
    def test_validation_model(self, df_train=pd.DataFrame(), df_test=pd.DataFrame(), apply_features=[],
                              model_load_path='models/', save_train_cols_path='', save_prob_path='',
                              scaler_switch='', fillna_value=-1):
        print ('---------------test validation------------------------')
        model = cPickle.load(open(model_load_path, 'rb'))  # load model path
        if apply_features == []:
            apply_features = self.__valid_columns
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        else:
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        print 'len(apply_features):', len(apply_features)
        y_train = df_train[[self.__label]].reset_index(drop=True)
        x_train = df_train[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        y_test = df_test[[self.__label]].reset_index(drop=True)
        x_test = df_test[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        if save_train_cols_path != '':  # save trainingset columns
            wf = open(save_train_cols_path, 'w')
            for col in x_train.columns:
                wf.write(col + '\n')
            wf.close()
        if scaler_switch == 'std':
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        elif scaler_switch == 'min_max':
            scaler = MinMaxScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
        X_train, Y_train, X_test, Y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        y_train, y_test = np.array(y_train), np.array(y_test)
        # 模型预测训练集得到概率，并保存起来
        y_predict = model.predict_proba(X_train)[:, 1]
        wf = open(save_prob_path, 'w')
        for item in y_predict:
            wf.write(str(item) + '\n')
        wf.close()
        # 模型预测测试集
        y_predict = model.predict_proba(X_test)[:, 1]
        settled_probs = read_settled_probs(save_prob_path)
        get_src_quantiles(settled_probs)
        src_quantiles = np.loadtxt('src_quantiles.txt')
        score = [calibrate_new(item, src_quantiles) for item in y_predict]  # 映射分数
        dict_test = ks(y_test, np.array(score))
        print "time validation AUC: %f" % metrics.roc_auc_score(Y_test, y_predict), "validation KS Score: %f" % \
                                                                                    dict_test['ks']
        print print_ks(dict_test)
        '''

    # 调用已有模型，根据已保存的概率值映射分数，并计算ks、分箱。该函数用于检验已经生成的模型文件表现是否与训练一致
    '''
    def model_file_test(self, df_test=pd.DataFrame(), apply_features=[], model_load_path='', prob_load_path='',
                        fillna_value=-1, scaler_switch=''):
        model = cPickle.load(open(model_load_path, 'rb'))  # load model path
        if apply_features == []:
            apply_features = self.__valid_columns
            apply_features.append(self.__label)
        else:
            if self.__label not in apply_features:
                apply_features.append(self.__label)
        print 'len(apply_features):', len(apply_features)
        y_test = df_test[[self.__label]].reset_index(drop=True)
        x_test = df_test[apply_features].drop([self.__label], axis=1).fillna(fillna_value).reset_index(drop=True)
        if scaler_switch == 'std':
            scaler = StandardScaler().fit(x_test)
            x_test = scaler.transform(x_test)
        elif scaler_switch == 'min_max':
            scaler = MinMaxScaler().fit(x_test)
            x_test = scaler.transform(x_test)
        X_test, Y_test = np.array(x_test), np.array(y_test)
        y_test = np.array(y_test)
        # 模型预测测试集
        y_predict = model.predict_proba(X_test)[:, 1]
        settled_probs = read_settled_probs(prob_load_path)
        get_src_quantiles(settled_probs)
        src_quantiles = np.loadtxt('src_quantiles.txt')
        score = [calibrate_new(item, src_quantiles) for item in y_predict]  # 映射分数
        dict_test = ks(y_test, np.array(score))
        print "time validation AUC: %f" % metrics.roc_auc_score(Y_test, y_predict), "validation KS Score: %f" % \
                                                                                    dict_test['ks']
        print print_ks(dict_test)
        '''
