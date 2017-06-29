import pandas as pd
import numpy as np
import pickle
from sklearn.svm import LinearSVR, SVR, NuSVR, SVC
from sklearn import ensemble
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

train_path = '../data/dengue_features_train.csv'
label_path = '../data/dengue_labels_train.csv'

def preprocess(feature_path,label_path):
    train_features = pd.read_csv(feature_path,index_col=[0,1])
    train_labels = pd.read_csv(label_path,index_col=[0,1,2])

    train_features.fillna(method='ffill', inplace=True)
    sj_train_features = train_features.loc['sj']
    sj_train_labels = train_labels.loc['sj']

    iq_train_features = train_features.loc['iq']
    iq_train_labels = train_labels.loc['iq']

    sj_train_features.drop('week_start_date', axis=1, inplace=True)
    #sj_train_features.drop('ndvi_se', axis=1, inplace=True)
    #sj_train_features.drop('ndvi_sw', axis=1, inplace=True)
    sj_train_features.drop('ndvi_ne', axis=1, inplace=True)
    #sj_train_features.drop('ndvi_nw', axis=1, inplace=True)
    iq_train_features.drop('week_start_date', axis=1, inplace=True)
    #iq_train_features.drop('ndvi_se', axis=1, inplace=True)
    #iq_train_features.drop('ndvi_sw', axis=1, inplace=True)
    #iq_train_features.drop('ndvi_ne', axis=1, inplace=True)
    #iq_train_features.drop('ndvi_nw', axis=1, inplace=True)

    return (sj_train_features.values, sj_train_labels.values),(iq_train_features.values,iq_train_labels.values)

def add_time(lag,train):
    new_train = np.array([np.append(train[j],train[j-lag:j]) for j in range(lag,len(train),1)])
    return new_train

def main():
    (sj_train,sj_label),(iq_train,iq_label) = preprocess(train_path,label_path)
    lag = 3
    s_mean = sj_train.mean(0)
    s_std = sj_train.std(0)
    np.save('../npy/s_mean',s_mean)
    np.save('../npy/s_std',s_std)
    i_mean = iq_train.mean(0)
    i_std = iq_train.std(0)
    np.save('../npy/i_mean',i_mean)
    np.save('../npy/i_std',i_std)

    sj_train = (sj_train-s_mean)/s_std
    iq_train = (iq_train-i_mean)/i_std

    sj_label = sj_label.reshape(-1,52)
    sj_label_mean = sj_label.mean(0)
    sj_label_std = sj_label.std(0)
    np.save('../npy/sj_label_mean',sj_label_mean)
    np.save('../npy/sj_label_std',sj_label_std)
    sj_label = (sj_label-sj_label_mean)/sj_label_std
    sj_label = sj_label.reshape(-1,1)

    iq_label = iq_label.reshape(-1,52)
    iq_label_mean = iq_label.mean(0)
    iq_label_std = iq_label.std(0)
    np.save('../npy/iq_label_mean',iq_label_mean)
    np.save('../npy/iq_label_std',iq_label_std)
    iq_label = (iq_label-iq_label_mean)/iq_label_std
    iq_label = iq_label.reshape(-1,1)

    sj_train = add_time(lag,sj_train)
    sj_label = sj_label[lag:]
    sj_label = sj_label.reshape(len(sj_label))
    iq_train = add_time(lag,iq_train)
    iq_label = iq_label[lag:]
    iq_label = iq_label.reshape(len(iq_label))
    # In sj train, we have 80 feature, and (33,27,38,29) are important

    print(sj_train.shape)
    length = sj_train.shape[0]
    #sj_train = np.delete(sj_train,[56, 36],1)  #mse ,mae
    #sj_train = np.delete(sj_train,[51, 74],1)  #mse ,mae
    #sj_train = np.delete(sj_train,[32, 24],1)  #mse
    print(sj_train.shape)
    print(sj_label.shape)
    print(iq_train.shape)
    print(iq_label.shape)
    
    #s_clf = ExtraTreesRegressor(n_estimators=2000, criterion='mae',max_depth=3)
    #s_clf = ensemble.BaggingRegressor(n_estimators=100,max_features=0.6,max_samples=0.6)
    s_clf = GradientBoostingRegressor(loss='huber', learning_rate=0.1, n_estimators=1000, max_depth=5, criterion='mae', verbose=1)
    #s_clf = xgb.XGBRegressor(n_estimators=250, subsample=0.4, max_depth=5, colsample_bytree=0.7,verbose=1)

    #i_clf = ensemble.BaggingRegressor(n_estimators=100,max_features=0.6,max_samples=0.6)
    i_clf = ExtraTreesRegressor(n_estimators=2000, criterion='mae',max_depth=3)
    #i_clf = xgb.XGBRegressor(n_estimators=200, subsample=0.6, max_depth=3, colsample_bytree=0.7,verbose=1)
    #scores_sj = cross_val_score(s_clf, sj_train, sj_label, cv=8, scoring='neg_mean_absolute_error',n_jobs=3,verbose=1)
    #scores_iq = cross_val_score(i_clf, iq_train, iq_label, cv=8, scoring='neg_mean_absolute_error',n_jobs=3,verbose=1)
    #print(scores_sj, scores_sj.mean(), scores_sj.std())
    #print(scores_iq, scores_iq.mean(), scores_iq.std())
    #s_clf = xgb.XGBRegressor(n_estimators=400, subsample=0.33, max_depth=5 ,silent=0, learning_rate=0.198)
    s_clf.fit(sj_train,sj_label)
    pickle.dump(s_clf,open('../model/s_clf_gb_try.pkl','wb'))
    i_clf.fit(iq_train,iq_label)
    pickle.dump(i_clf,open('../model/i_clf_xtree_try.pkl','wb'))
    #pickle.dump(s_clf,open('s_clf_bag.pkl','wb'))

if __name__=='__main__':
    main()
