import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from datetime import datetime
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt
from matplotlib import cm
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import BatchNormalization, Input, Embedding, Concatenate, Conv1D, MaxPooling1D, Flatten
from keras.models import Sequential, Model
import keras.backend as K
import statsmodels.formula.api as smf
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


# ---------------------------------------- GLOBAL PARAMETERS

NUM_LAGS = 10
sel = [0,2,4,5,7,8,9] # weather features to use
sel = [5,7]
sel2 = [0,1,2,7] # eventlags featurs to use

# word embeddings parameters
#GLOVE_DIR = "/home/fmpr/datasets/glove.6B/"
GLOVE_DIR = "/mnt/sdb1/datasets/glove.6B/"
MAX_SEQUENCE_LENGTH = 350 #600
MAX_NB_WORDS = 600 #5000
EMBEDDING_DIM = 300 #300


# ---------------------------------------- Load weather data
print "loading weather data..."

# load data
df = pd.read_csv("../central_park_weather.csv")
df = df.set_index("date")

# replace predefined values with NaN
df = df.replace(99.99, np.nan)
df = df.replace(999.9, np.nan)
df = df.replace(9999.9, np.nan)

# replace NaN with 0 for snow depth
df["snow_depth"] = df["snow_depth"].fillna(0)

# do interpolation for the remaining NaNs
df = df.interpolate()

# standardize data
removed_mean = df.mean()
removed_std = df.std()
weather = (df - removed_mean) / removed_std


# ---------------------------------------- Load events data
print "loading events data..."

events = pd.read_csv("terminal5_events_preprocessed.tsv", sep="\t")
events.head()

events['start_time'] = pd.to_datetime(events['start_time'], format='%Y-%m-%d %H:%M')
events['date'] = events['start_time'].dt.strftime("%Y-%m-%d")
events = events[["date","start_time","title","url","description"]]


# ---------------------------------------- Load taxi data (and merge with others and detrend)
print "loading taxi data (and merging and detrending)..."

df = pd.read_csv("/home/fmpr/data/pickups_terminal_5_0.003.csv")

df_sum = pd.DataFrame(df.groupby("date")["pickups"].sum())
df_sum["date"] = df_sum.index
df_sum.index = pd.to_datetime(df_sum.index, format='%Y-%m-%d %H:%M')
df_sum["dow"] = df_sum.index.weekday

# add events information
event_col = np.zeros((len(df_sum)))
late_event = np.zeros((len(df_sum)))
really_late_event = np.zeros((len(df_sum)))
event_desc_col = []
for i in xrange(len(df_sum)):
    if df_sum.iloc[i].date in events["date"].values:
        event_col[i] = 1
        event_descr = ""
        for e in events[events.date == df_sum.iloc[i].date]["description"]:
            event_descr += str(e) + " "
        event_desc_col.append(event_descr)
        for e in events[events.date == df_sum.iloc[i].date]["start_time"]:
            if e.hour >= 20:
                late_event[i] = 1
            if e.hour >= 21:
                really_late_event[i] = 1
    else:
        event_desc_col.append("None")

df_sum["event"] = event_col
df_sum["late_event"] = late_event
df_sum["really_late_event"] = really_late_event
df_sum["event_desc"] = event_desc_col
df_sum["event_next_day"] = pd.Series(df_sum["event"]).shift(-1)
df_sum["late_event_next_day"] = pd.Series(df_sum["late_event"]).shift(-1)
df_sum["really_late_event_next_day"] = pd.Series(df_sum["really_late_event"]).shift(-1)
df_sum["event_next_day_desc"] = pd.Series(df_sum["event_desc"]).shift(-1)

# merge with weather data
df_sum = df_sum.join(weather, how='inner')
df_sum.head()

# keep only data after 2013
START_YEAR = 2013
df_sum = df_sum.loc[df_sum.index.year >= START_YEAR]
df_sum.head()

df_sum["year"] = df_sum.index.year

trend_mean = df_sum[df_sum.index.year < 2015].groupby(["dow"]).mean()["pickups"]

#trend_std = df_sum.groupby(["year"]).std()["pickups"]
trend_std = df_sum["pickups"].std()

# build vectors with trend to remove and std
trend = []
std = []
for ix, row in df_sum.iterrows():
    trend.append(trend_mean[row.dow])
    #std.append(trend_std[row.year])
    std.append(trend_std)

df_sum["trend"] = trend
df_sum["std"] = std

# detrend data
df_sum["detrended"] = (df_sum["pickups"] - df_sum["trend"]) / df_sum["std"]


# ---------------------------------------- Build lags and features
print "building lags..."

lags = pd.concat([pd.Series(df_sum["detrended"]).shift(x) for x in range(0,NUM_LAGS)],axis=1).as_matrix()
event_feats = np.concatenate([df_sum["event_next_day"].as_matrix()[:,np.newaxis],
                             df_sum["late_event"].as_matrix()[:,np.newaxis],
                             #df_sum["late_event_next_day"].as_matrix()[:,np.newaxis],
                             df_sum["really_late_event"].as_matrix()[:,np.newaxis],
                             df_sum["really_late_event_next_day"].as_matrix()[:,np.newaxis]], axis=1)
lags_event_feats = pd.concat([pd.Series(df_sum["event_next_day"]).shift(x) for x in range(0,NUM_LAGS)],axis=1).as_matrix()
event_texts = df_sum["event_next_day_desc"].as_matrix()
weather_feats = df_sum[['min_temp', u'max_temp', u'wind_speed',
       u'wind_gust', u'visibility', u'pressure', u'precipitation',
       u'snow_depth', u'fog', u'rain_drizzle', u'snow_ice', u'thunder']].as_matrix()
preds = pd.Series(df_sum["detrended"]).shift(-1).as_matrix()
trends = df_sum["trend"].as_matrix()
stds = df_sum["std"].as_matrix()

lags = lags[NUM_LAGS:-1,:]
event_feats = event_feats[NUM_LAGS:-1,:]
lags_event_feats = lags_event_feats[NUM_LAGS:-1,:]
event_texts = event_texts[NUM_LAGS:-1]
weather_feats = weather_feats[NUM_LAGS:-1,:]
preds = preds[NUM_LAGS:-1]
trends = trends[NUM_LAGS:-1]
stds = stds[NUM_LAGS:-1]


# ---------------------------------------- Train/test split
print "loading train/val/test split..."

#i_train = 365*2-90 # 2013 and 2014
#i_val = 365*2
i_train = 365*2 # 2013 and 2014
i_val = 365*3
i_test = -1 # 2015 and 2016 (everything else)

lags_train = lags[:i_train,:] # time series lags
event_feats_train = event_feats[:i_train,:] # event/no_event
lags_event_feats_train = lags_event_feats[:i_train,:] # lags for event/no_event
event_texts_train = event_texts[:i_train] # event text descriptions
weather_feats_train = weather_feats[:i_train,:] # weather data
y_train = preds[:i_train] # target values

lags_val = lags[i_train:i_val,:] # time series lags
event_feats_val = event_feats[i_train:i_val,:] # event/no_event
lags_event_feats_val = lags_event_feats[i_train:i_val,:] # lags for event/no_event
event_texts_val = event_texts[i_train:i_val] # event text descriptions
weather_feats_val = weather_feats[i_train:i_val,:] # weather data
y_val = preds[i_train:i_val] # target values

lags_test = lags[i_val:i_test,:]
event_feats_test = event_feats[i_val:i_test,:]
lags_event_feats_test = lags_event_feats[i_val:i_test,:]
event_texts_test = event_texts[i_val:i_test]
weather_feats_test = weather_feats[i_val:i_test,:]
y_test = preds[i_val:i_test]
trend_test = trends[i_val:i_test]
std_test = stds[i_val:i_test]


# ---------------------------------------- Evaluation functions

def compute_error(trues, predicted):
    corr = np.corrcoef(predicted, trues)[0,1]
    mae = np.mean(np.abs(predicted - trues))
    rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    rrse = np.sqrt(np.sum((predicted - trues)**2) / np.sum((trues - np.mean(trues))**2))
    mape = np.mean(np.abs((predicted - trues) / trues)) * 100
    r2 = max(0, 1 - np.sum((predicted - trues)**2) / np.sum((trues - np.mean(trues))**2))
    return corr, mae, rae, rmse, rrse, mape, r2


def compute_error_filtered(trues, predicted, filt):
    trues = trues[filt]
    predicted = predicted[filt]
    corr = np.corrcoef(predicted, trues)[0,1]
    mae = np.mean(np.abs(predicted - trues))
    mse = np.mean((predicted - trues)**2)
    rae = np.sum(np.abs(predicted - trues)) / np.sum(np.abs(trues - np.mean(trues)))
    rmse = np.sqrt(np.mean((predicted - trues)**2))
    r2 = max(0, 1 - np.sum((trues-predicted)**2) / np.sum((trues - np.mean(trues))**2))
    return corr, mae, rae, rmse, rrse, mape, r2

# ---------------------------------------- Output files

if not os.path.exists("results_mae.txt"):
    fw_mae = open("results_mae.txt", "a")
    fw_mae.write("HA,ARIMA,SVR L,SVR L+W,SVR L+W+E,SVR L+W+E+LF,SVR L+W+E+LF+EL,")
    fw_mae.write("GP L,GP L+W,GP L+W+E,GP L+W+E+LF,GP L+W+E+LF+EL\n")
    #fw_mae.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_rae = open("results_rae.txt", "a")
    fw_rae.write("HA,ARIMA,SVR L,SVR L+W,SVR L+W+E,SVR L+W+E+LF,SVR L+W+E+LF+EL,")
    fw_rae.write("GP L,GP L+W,GP L+W+E,GP L+W+E+LF,GP L+W+E+LF+EL\n")
    #fw_rae.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_rmse = open("results_rmse.txt", "a")
    fw_rmse.write("HA,ARIMA,SVR L,SVR L+W,SVR L+W+E,SVR L+W+E+LF,SVR L+W+E+LF+EL,")
    fw_rmse.write("GP L,GP L+W,GP L+W+E,GP L+W+E+LF,GP L+W+E+LF+EL\n")
    #fw_rmse.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_rrse = open("results_rrse.txt", "a")
    fw_rrse.write("HA,ARIMA,SVR L,SVR L+W,SVR L+W+E,SVR L+W+E+LF,SVR L+W+E+LF+EL,")
    fw_rrse.write("GP L,GP L+W,GP L+W+E,GP L+W+E+LF,GP L+W+E+LF+EL\n")
    #fw_rrse.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_mape = open("results_mape.txt", "a")
    fw_mape.write("HA,ARIMA,SVR L,SVR L+W,SVR L+W+E,SVR L+W+E+LF,SVR L+W+E+LF+EL,")
    fw_mape.write("GP L,GP L+W,GP L+W+E,GP L+W+E+LF,GP L+W+E+LF+EL\n")
    #fw_mape.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_r2 = open("results_r2.txt", "a")
    fw_r2.write("HA,ARIMA,SVR L,SVR L+W,SVR L+W+E,SVR L+W+E+LF,SVR L+W+E+LF+EL,")
    fw_r2.write("GP L,GP L+W,GP L+W+E,GP L+W+E+LF,GP L+W+E+LF+EL\n")
    #fw_r2.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
else:
    fw_mae = open("results_mae.txt", "a")
    fw_rae = open("results_rae.txt", "a")
    fw_rmse = open("results_rmse.txt", "a")
    fw_rrse = open("results_rrse.txt", "a")
    fw_mape = open("results_mape.txt", "a")
    fw_r2 = open("results_r2.txt", "a")


from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, RationalQuadratic
from statsmodels.tsa.arima_model import ARIMA

# Historical averages model
print "\nrunning historical averages model:"
preds_lr = trend_test
y_true = y_test * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lr)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))

# ARIMA model
print "\nrunning ARIMA model"
history = [x for x in y_train]
predictions = []
print len(y_test)
for t in range(len(y_test)):
    #break
    #model = ARIMA(history, order=(5,1,0)) # MAE:  167.404   RMSE: 213.824   R2:   0.000
    #model = ARIMA(history, order=(5,0,0)) # MAE:  165.829   RMSE: 213.664   R2:   0.000
    #model = ARIMA(history, order=(1,0,0)) # MAE:  182.480   RMSE: 236.725   R2:   0.000
    #model = ARIMA(history, order=(1,0,1)) # MAE:  156.624   RMSE: 200.244   R2:   0.000
    #model = ARIMA(history, order=(1,1,1)) # MAE:  159.184   RMSE: 202.252   R2:   0.000
    #model = ARIMA(history, order=(2,0,1)) # MAE:  119.335   RMSE: 161.513   R2:   0.468
    model = ARIMA(history, order=(2,1,1)) # MAE:  120.909   RMSE: 161.509   R2:   0.468
    #model = ARIMA(history, order=(3,1,1)) # MAE:  120.092   RMSE: 160.780   R2:   0.473
    model_fit = model.fit(disp=0)
    try:
        #model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0][0]
        #print yhat
    except:
        print "ERROR fitting ARIMA"
        #print prev_obs
        yhat = prev_obs
    
    predictions.append(yhat)
    obs = y_test[t]
    #print obs
    prev_obs = obs
    history.append(obs)
    print t, " MAE:", np.mean(np.abs(np.array(predictions) - np.array(y_test[:t+1])))
    #print('[%d] predicted=%.2f, expected=%.2f, error=%.2f' % (t, yhat, obs, np.abs(np.array(yhat)-y_test[t])))

#model = ARIMA(np.concatenate([y_train, y_test]), order=(10,0,0)) #
#model_fit = model.fit(disp=0)
#predictions = model_fit.predict(start=len(y_train))

#predictions = trend_test

preds_lr = np.array(predictions) * std_test + trend_test
y_true = y_test * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lr)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))

for C in [0.01,0.1,1,10,100]:
    for epsilon in []:
    #for epsilon in [0.01,0.1,1,10,100]:
        #neigh = KNeighborsRegressor(n_neighbors=num_neighbors)
        #neigh = SVR(C=C, epsilon=epsilon)
        neigh = SVR(kernel='rbf', C=C, gamma=epsilon)
        #neigh = SVR(kernel='linear', C=C)
        #neigh = SVR(kernel='poly', C=C, degree=epsilon)
        
        #kernel = 1.0 * Matern(length_scale=gp_lenscale, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=gp_noise, noise_level_bounds=(1e-10, 1e+2))
        #kernel = 1.0 * RationalQuadratic(length_scale=gp_lenscale, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=gp_noise, noise_level_bounds=(1e-10, 1e+2))
        #neigh = GaussianProcessRegressor(kernel=kernel, alpha=1.)
        #neigh.fit(lags_train, y_train)
        #preds_lr = neigh.predict(lags_test)
        neigh.fit(np.concatenate([lags_train, event_feats_train, lags_event_feats_train[:,sel2], weather_feats_train[:,sel]], axis=1), y_train)
        preds_lr = neigh.predict(np.concatenate([lags_test, event_feats_test, lags_event_feats_test[:,sel2], weather_feats_test[:,sel]], axis=1))
        preds_lr = preds_lr * std_test + trend_test
        y_true = y_test * std_test + trend_test
        corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lr)
        #print gp_lenscale, gp_noise
        print C, epsilon
        print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)

C = 0.1
epsilon = 1.0

# for linear kernel
C = 100.0

# for Guassian kernel
C = 10.0
epsilon = 0.001


# ---------------------------------------- KNN baseline (just lags)

# KNN (just lags)
print "\nrunning SVR with just lags..."
neigh = SVR(kernel='rbf', C=C, gamma=epsilon)
neigh = SVR(kernel='linear', C=C)
neigh.fit(lags_train, y_train)
preds_lr = neigh.predict(lags_test)
preds_lr = preds_lr * std_test + trend_test
y_true = y_test * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lr)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))


# ---------------------------------------- KNN with weather

# KNN lags + weather
print "\nrunning SVR with lags + weather..."
neigh = SVR(kernel='rbf', C=C, gamma=epsilon)
neigh = SVR(kernel='linear', C=C)
neigh.fit(np.concatenate([lags_train, weather_feats_train[:,sel]], axis=1), y_train)
preds_lr = neigh.predict(np.concatenate([lags_test, weather_feats_test[:,sel]], axis=1))
preds_lr = preds_lr * std_test + trend_test
y_true = y_test * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lr)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))


# ---------------------------------------- KNN with weather + events

# KNN lags + weather + events
print "\nrunning SVR with lags + weather + events..."
neigh = SVR(kernel='rbf', C=C, gamma=epsilon)
neigh = SVR(kernel='linear', C=C)
neigh.fit(np.concatenate([lags_train, weather_feats_train[:,sel], event_feats_train[:,:1]], axis=1), y_train)
preds_lr = neigh.predict(np.concatenate([lags_test, weather_feats_test[:,sel], event_feats_test[:,:1]], axis=1))
preds_lr = preds_lr * std_test + trend_test
y_true = y_test * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lr)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))

# KNN lags + weather + events + late
print "\nrunning SVR with lags + weather + late..."
neigh = SVR(kernel='rbf', C=C, gamma=epsilon)
neigh = SVR(kernel='linear', C=C)
neigh.fit(np.concatenate([lags_train, weather_feats_train[:,sel], event_feats_train], axis=1), y_train)
preds_lr = neigh.predict(np.concatenate([lags_test, weather_feats_test[:,sel], event_feats_test], axis=1))
preds_lr = preds_lr * std_test + trend_test
y_true = y_test * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lr)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))

# KNN lags + weather + events + late + event_lags
print "\nrunning SVR with lags + weather + late + event_lags..."
neigh = SVR(kernel='rbf', C=C, gamma=epsilon)
neigh = SVR(kernel='linear', C=C)
neigh.fit(np.concatenate([lags_train, event_feats_train, lags_event_feats_train[:,sel2], weather_feats_train[:,sel]], axis=1), y_train)
preds_lr = neigh.predict(np.concatenate([lags_test, event_feats_test, lags_event_feats_test[:,sel2], weather_feats_test[:,sel]], axis=1))
preds_lr = preds_lr * std_test + trend_test
y_true = y_test * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lr)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))


gp_lenscale = 1.0
gp_scale = 1.0
gp_noise = 1e-5
gp_alpha = 0.0

# for RBF
gp_lenscale = 10.0
gp_scale = 1.0
gp_noise = 1.0
gp_alpha = 0.8

# for matern
gp_lenscale = 10.0
gp_scale = 1.0
gp_noise = 1.0
gp_alpha = 0.8


# ---------------------------------------- GP baseline (just lags)

# GP (just lags)
print "\nrunning GP with just lags..."
#kernel = gp_scale * RBF(length_scale=gp_lenscale, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=gp_noise, noise_level_bounds=(1e-10, 1e+1))
kernel = gp_scale * Matern(length_scale=gp_lenscale, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=gp_noise, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=gp_alpha)
gp.fit(lags_train, y_train)
preds_lr = gp.predict(lags_test)
preds_lr = preds_lr * std_test + trend_test
y_true = y_test * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lr)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))


# ---------------------------------------- GP with weather

# GP lags + weather
print "\nrunning GP with lags + weather..."
kernel = gp_scale * RBF(length_scale=gp_lenscale, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=gp_noise, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=gp_alpha)
gp.fit(np.concatenate([lags_train, weather_feats_train[:,sel]], axis=1), y_train)
preds_lr = gp.predict(np.concatenate([lags_test, weather_feats_test[:,sel]], axis=1))
preds_lr = preds_lr * std_test + trend_test
y_true = y_test * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lr)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))


# ---------------------------------------- GP with weather + events

# GP lags + weather + events
print "\nrunning GP with lags + weather + events..."
kernel = gp_scale * RBF(length_scale=gp_lenscale, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=gp_noise, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=gp_alpha)
gp.fit(np.concatenate([lags_train, weather_feats_train[:,sel], event_feats_train[:,:1]], axis=1), y_train)
preds_lr = gp.predict(np.concatenate([lags_test, weather_feats_test[:,sel], event_feats_test[:,:1]], axis=1))
preds_lr = preds_lr * std_test + trend_test
y_true = y_test * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lr)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))

# GP lags + weather + events + late
print "\nrunning GP with lags + weather + late..."
kernel = gp_scale * RBF(length_scale=gp_lenscale, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=gp_noise, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=gp_alpha)
gp.fit(np.concatenate([lags_train, weather_feats_train[:,sel], event_feats_train], axis=1), y_train)
preds_lr = gp.predict(np.concatenate([lags_test, weather_feats_test[:,sel], event_feats_test], axis=1))
preds_lr = preds_lr * std_test + trend_test
y_true = y_test * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lr)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))

# GP lags + weather + events + late + event_lags
print "\nrunning GP with lags + weather + late + event_lags..."
kernel = gp_scale * RBF(length_scale=gp_lenscale, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=gp_noise, noise_level_bounds=(1e-10, 1e+1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=gp_alpha)
gp.fit(np.concatenate([lags_train, event_feats_train, lags_event_feats_train[:,sel2], weather_feats_train[:,sel]], axis=1), y_train)
preds_lr = gp.predict(np.concatenate([lags_test, event_feats_test, lags_event_feats_test[:,sel2], weather_feats_test[:,sel]], axis=1))
preds_lr = preds_lr * std_test + trend_test
y_true = y_test * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lr)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))



# close output files
fw_mae.write("\n")
fw_rae.write("\n")
fw_rmse.write("\n")
fw_rrse.write("\n")
fw_mape.write("\n")
fw_r2.write("\n")
fw_mae.close()
fw_rae.close()
fw_rmse.close()
fw_rrse.close()
fw_mape.close()
fw_r2.close()


