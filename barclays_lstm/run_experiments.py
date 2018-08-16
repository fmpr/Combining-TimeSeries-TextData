import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

# word embeddings parameters
GLOVE_DIR = "/home/fmpr/datasets/glove.6B/"
#GLOVE_DIR = "/mnt/sdb1/datasets/glove.6B/"
MAX_SEQUENCE_LENGTH = 600 #1000
MAX_NB_WORDS = 5000 #20000
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

events = pd.read_csv("barclays_events_preprocessed.tsv", sep="\t")
events.head()

events['start_time'] = pd.to_datetime(events['start_time'], format='%Y-%m-%d %H:%M')
events['date'] = events['start_time'].dt.strftime("%Y-%m-%d")
events = events[["date","start_time","title","url","description"]]


# ---------------------------------------- Load taxi data (and merge with others and detrend)
print "loading taxi data (and merging and detrending)..."

df = pd.read_csv("/home/fmpr/data/pickups_barclays_center_0.003.csv")

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

i_train = 365*2-90 # 2013 and 2014
i_val = 365*2
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
    return corr, mae, mse, rae, rmse, r2

# ---------------------------------------- Output files

if not os.path.exists("results_mae.txt"):
    fw_mae = open("results_mae.txt", "a")
    fw_mae.write("LR L,LR L+W,LR L+W+E,LR L+W+E+LF,LR L+W+E+LF+EL,")
    fw_mae.write("MLP L,MLP L+W,MLP L+W+E,MLP L+W+E+LF,MLP L+W+E+LF+EL,")
    fw_mae.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_rae = open("results_rae.txt", "a")
    fw_rae.write("LR L,LR L+W,LR L+W+E,LR L+W+E+LF,LR L+W+E+LF+EL,")
    fw_rae.write("MLP L,MLP L+W,MLP L+W+E,MLP L+W+E+LF,MLP L+W+E+LF+EL,")
    fw_rae.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_rmse = open("results_rmse.txt", "a")
    fw_rmse.write("LR L,LR L+W,LR L+W+E,LR L+W+E+LF,LR L+W+E+LF+EL,")
    fw_rmse.write("MLP L,MLP L+W,MLP L+W+E,MLP L+W+E+LF,MLP L+W+E+LF+EL,")
    fw_rmse.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_rrse = open("results_rrse.txt", "a")
    fw_rrse.write("LR L,LR L+W,LR L+W+E,LR L+W+E+LF,LR L+W+E+LF+EL,")
    fw_rrse.write("MLP L,MLP L+W,MLP L+W+E,MLP L+W+E+LF,MLP L+W+E+LF+EL,")
    fw_rrse.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_mape = open("results_mape.txt", "a")
    fw_mape.write("LR L,LR L+W,LR L+W+E,LR L+W+E+LF,LR L+W+E+LF+EL,")
    fw_mape.write("MLP L,MLP L+W,MLP L+W+E,MLP L+W+E+LF,MLP L+W+E+LF+EL,")
    fw_mape.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
    fw_r2 = open("results_r2.txt", "a")
    fw_r2.write("LR L,LR L+W,LR L+W+E,LR L+W+E+LF,LR L+W+E+LF+EL,")
    fw_r2.write("MLP L,MLP L+W,MLP L+W+E,MLP L+W+E+LF,MLP L+W+E+LF+EL,")
    fw_r2.write("MLP L+W+E+LF+ET,MLP L+W+E+LF+EL+ET\n")
else:
    fw_mae = open("results_mae.txt", "a")
    fw_rae = open("results_rae.txt", "a")
    fw_rmse = open("results_rmse.txt", "a")
    fw_rrse = open("results_rrse.txt", "a")
    fw_mape = open("results_mape.txt", "a")
    fw_r2 = open("results_r2.txt", "a")


# ---------------------------------------- Linear regression baseline (just lags)

# linear regression (just lags)
print "\nrunning linear regression with just lags..."
regr = linear_model.LinearRegression()
regr.fit(lags_train, y_train)
preds_lr = regr.predict(lags_test)
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


# ---------------------------------------- Linear regression with weather

# linear regression lags + weather
print "\nrunning linear regression with lags + weather..."
regr = linear_model.LinearRegression()
regr.fit(np.concatenate([lags_train, weather_feats_train[:,sel]], axis=1), y_train)
preds_lr = regr.predict(np.concatenate([lags_test, weather_feats_test[:,sel]], axis=1))
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


# ---------------------------------------- Linear regression with weather + events

# linear regression lags + weather + events
print "\nrunning linear regression with lags + weather + events..."
regr = linear_model.LinearRegression()
regr.fit(np.concatenate([lags_train, weather_feats_train[:,sel], event_feats_train[:,:1]], axis=1), y_train)
preds_lr = regr.predict(np.concatenate([lags_test, weather_feats_test[:,sel], event_feats_test[:,:1]], axis=1))
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

# linear regression lags + weather + events + late
print "\nrunning linear regression with lags + weather + late..."
regr = linear_model.LinearRegression()
regr.fit(np.concatenate([lags_train, weather_feats_train[:,sel], event_feats_train], axis=1), y_train)
preds_lr = regr.predict(np.concatenate([lags_test, weather_feats_test[:,sel], event_feats_test], axis=1))
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

# linear regression lags + weather + events + late + event_lags
print "\nrunning linear regression with lags + weather + late + event_lags..."
regr = linear_model.LinearRegression()
regr.fit(np.concatenate([lags_train, event_feats_train, lags_event_feats_train, weather_feats_train[:,sel]], axis=1), y_train)
preds_lr = regr.predict(np.concatenate([lags_test, event_feats_test, lags_event_feats_test, weather_feats_test[:,sel]], axis=1))
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


# ---------------------------------------- LSTM (just lags)

def build_model(num_inputs, num_lags, num_preds):
    input_lags = Input(shape=(num_lags,1))
    
    x = input_lags
    #x = BatchNormalization()(x)
    x = LSTM(30, 
             kernel_regularizer=keras.regularizers.l2(0.2), 
             recurrent_regularizer=keras.regularizers.l2(0.2), 
             return_sequences=False)(x)
    #x = Activation("relu")(x)
    #x = Flatten()(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.2)(x)
    #x = BatchNormalization()(x)
    
    #preds = Dense(units=num_preds)(x)
    preds = Dense(units=num_preds, kernel_regularizer=keras.regularizers.l2(0.2))(x)
    
    model = Model(input_lags, preds)
    
    #rmsp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss="mse", optimizer="adam")
    
    return model, input_lags, preds


print "\nrunning LSTM with just lags..."

# checkpoint best model
checkpoint = ModelCheckpoint("weights.best.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')

model, input_lags, preds = build_model(1, NUM_LAGS, 1)
model.fit(
    lags_train[:,:,np.newaxis],
    y_train,
    batch_size=64,
    epochs=500,
    validation_data=(lags_val[:,:,np.newaxis], y_val),
    callbacks=[checkpoint],
    verbose=0)   

print "Total number of iterations:  ", len(model.history.history["loss"])
print "Best loss at iteratation:    ", np.argmin(model.history.history["loss"]), "   Best:", np.min(model.history.history["loss"])
print "Best val_loss at iteratation:", np.argmin(model.history.history["val_loss"]), "   Best:", np.min(model.history.history["val_loss"])

# load weights
model.load_weights("weights.best.hdf5")

# make predictions
preds_lstm = model.predict(np.concatenate([lags_test[:,:,np.newaxis]], axis=1))
preds_lstm = preds_lstm[:,0] * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lstm)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))


# ---------------------------------------- LSTM lags + weather

def build_model_weather(num_inputs, num_lags, num_feat, num_preds):
    input_lags = Input(shape=(num_lags,1))
    input_weather = Input(shape=(num_feat,))
    
    x_lags = input_lags
    #x_lags = BatchNormalization()(x_lags)
    x_lags = LSTM(30, 
             kernel_regularizer=keras.regularizers.l2(0.2), 
             recurrent_regularizer=keras.regularizers.l2(0.2), 
             return_sequences=False)(x_lags)
    #x_lags = Activation("relu")(x_lags)
    #x_lags = Flatten()(x_lags)
    x_lags = BatchNormalization()(x_lags)
    #x_lags = Dropout(0.2)(x_lags)
    #x_lags = BatchNormalization()(x_lags)
    
    feat = Concatenate(axis=1)([x_lags, input_weather])
    
    feat = BatchNormalization()(feat)
    #preds = Dense(units=num_preds)(feat)
    preds = Dense(units=num_preds, kernel_regularizer=keras.regularizers.l2(0.2))(feat)
    preds = Activation("linear")(preds)
    
    model = Model([input_lags, input_weather], preds)
    
    rmsp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    #model.compile(loss="mse", optimizer=rmsp)
    model.compile(loss="mse", optimizer="adam")
    
    return model, input_lags, preds

print "\nrunning LSTM with lags + weather..."

# checkpoint best model
checkpoint = ModelCheckpoint("weights.best.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')

model, input_lags, preds = build_model_weather(1, NUM_LAGS, len(sel), 1)
model.fit(
    #lags_train, 
    #np.concatenate([lags_train, weather_feats_train[:,sel]], axis=1),
    [lags_train[:,:,np.newaxis], weather_feats_train[:,sel]],
    y_train,
    batch_size=64,
    epochs=500,
    #validation_data=(lags_val, y_val),
    #validation_data=(np.concatenate([lags_val, weather_feats_val[:,sel]], axis=1), y_val),
    validation_data=([lags_val[:,:,np.newaxis], weather_feats_val[:,sel]], y_val),
    callbacks=[checkpoint],
    verbose=0)   

print "Total number of iterations:  ", len(model.history.history["loss"])
print "Best loss at iteratation:    ", np.argmin(model.history.history["loss"]), "   Best:", np.min(model.history.history["loss"])
print "Best val_loss at iteratation:", np.argmin(model.history.history["val_loss"]), "   Best:", np.min(model.history.history["val_loss"])

# load weights
model.load_weights("weights.best.hdf5")

# make predictions
preds_lstm = model.predict([lags_test[:,:,np.newaxis], weather_feats_test[:,sel]])
preds_lstm = preds_lstm[:,0] * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lstm)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))


# ---------------------------------------- LSTM with weather + events information (no text)

def build_model_events(num_inputs, num_lags, num_feat, num_preds):
    input_lags = Input(shape=(num_lags,1))
    input_events = Input(shape=(num_lags,1))
    input_events2 = Input(shape=(num_feat,))
    
    x_lags = Concatenate(axis=2)([input_lags, input_events])
    #x_lags = BatchNormalization()(x_lags)
    x_lags = LSTM(30, 
             kernel_regularizer=keras.regularizers.l2(0.2), 
             recurrent_regularizer=keras.regularizers.l2(0.2), 
             return_sequences=False)(x_lags)
    #x_lags = Activation("relu")(x_lags)
    #x_lags = Flatten()(x_lags)
    x_lags = BatchNormalization()(x_lags)
    #x_lags = Dropout(0.2)(x_lags)
    #x_lags = BatchNormalization()(x_lags)
    
    feat = Concatenate(axis=1)([x_lags, input_events2])
    
    feat = BatchNormalization()(feat)
    #preds = Dense(units=num_preds)(feat)
    preds = Dense(units=num_preds, kernel_regularizer=keras.regularizers.l2(0.2))(feat)
    preds = Activation("linear")(preds)
    
    model = Model([input_lags, input_events, input_events2], preds)
    
    rmsp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    #model.compile(loss="mse", optimizer=rmsp)
    model.compile(loss="mse", optimizer="adam")
    
    return model, input_lags, preds


print "\nrunning LSTM with lags + weather + event..."

# checkpoint best model
checkpoint = ModelCheckpoint("weights.best.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')

# fit model to the mean
model, input_lags, preds = build_model_events(1, NUM_LAGS, len(sel)+1, 1)
#print model_weather.summary()
model.fit(
    #[np.concatenate([lags_train, weather_feats_train[:,sel]], axis=1), 
    # np.concatenate([event_feats_train[:,:1]], axis=1)],
    [lags_train[:,:,np.newaxis], lags_event_feats_train[:,:,np.newaxis], np.concatenate([weather_feats_train[:,sel], event_feats_train[:,:1]], axis=1)],
    y_train,
    batch_size=64,
    epochs=500,
    #validation_split=0.2,
    #validation_data=([np.concatenate([lags_val, weather_feats_val[:,sel]], axis=1), 
    #                  np.concatenate([event_feats_val[:,:1]], axis=1)], y_val),
    validation_data=([lags_val[:,:,np.newaxis], lags_event_feats_val[:,:,np.newaxis], np.concatenate([weather_feats_val[:,sel], event_feats_val[:,:1]], axis=1)], y_val),
    callbacks=[checkpoint],
    verbose=0)     

print "Total number of iterations:  ", len(model.history.history["loss"])
print "Best loss at iteratation:    ", np.argmin(model.history.history["loss"]), "   Best:", np.min(model.history.history["loss"])
print "Best val_loss at iteratation:", np.argmin(model.history.history["val_loss"]), "   Best:", np.min(model.history.history["val_loss"])

# load weights
model.load_weights("weights.best.hdf5")

print model.evaluate([lags_test[:,:,np.newaxis], lags_event_feats_test[:,:,np.newaxis], np.concatenate([weather_feats_test[:,sel], event_feats_test[:,:1]], axis=1)], 
                      y_test, verbose=2)

# make predictions
preds_lstm = model.predict([lags_test[:,:,np.newaxis], lags_event_feats_test[:,:,np.newaxis], np.concatenate([weather_feats_test[:,sel], event_feats_test[:,:1]], axis=1)])
preds_lstm = preds_lstm[:,0] * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lstm)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))


# ---------------------------------------- LSTM with weather + events information (no text) + late

print "\nrunning LSTM with lags + weather + event + late..."

# checkpoint best model
checkpoint = ModelCheckpoint("weights.best.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')

# fit model to the mean
model, input_lags, preds = build_model_events(1, NUM_LAGS, len(sel)+4, 1)
#print model_weather.summary()
model.fit(
    #[np.concatenate([lags_train, weather_feats_train[:,sel]], axis=1), 
    # np.concatenate([event_feats_train[:,:1]], axis=1)],
    [lags_train[:,:,np.newaxis], lags_event_feats_train[:,:,np.newaxis], np.concatenate([weather_feats_train[:,sel], event_feats_train], axis=1)],
    y_train,
    batch_size=64,
    epochs=500,
    #validation_split=0.2,
    #validation_data=([np.concatenate([lags_val, weather_feats_val[:,sel]], axis=1), 
    #                  np.concatenate([event_feats_val[:,:1]], axis=1)], y_val),
    validation_data=([lags_val[:,:,np.newaxis], lags_event_feats_val[:,:,np.newaxis], np.concatenate([weather_feats_val[:,sel], event_feats_val], axis=1)], y_val),
    callbacks=[checkpoint],
    verbose=0)   


print "Total number of iterations:  ", len(model.history.history["loss"])
print "Best loss at iteratation:    ", np.argmin(model.history.history["loss"]), "   Best:", np.min(model.history.history["loss"])
print "Best val_loss at iteratation:", np.argmin(model.history.history["val_loss"]), "   Best:", np.min(model.history.history["val_loss"])

# load weights
model.load_weights("weights.best.hdf5")

print model.evaluate([lags_test[:,:,np.newaxis], lags_event_feats_test[:,:,np.newaxis], np.concatenate([weather_feats_test[:,sel], event_feats_test], axis=1)], 
                      y_test, verbose=2)

# make predictions
preds_lstm = model.predict([lags_test[:,:,np.newaxis], lags_event_feats_test[:,:,np.newaxis], np.concatenate([weather_feats_test[:,sel], event_feats_test], axis=1)])
preds_lstm = preds_lstm[:,0] * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lstm)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))


# ---------------------------------------- LSTM with weather + events information (no text) + late + event_lags


print "\nrunning LSTM with lags + weather + event + late + envet_lags..."

# checkpoint best model
checkpoint = ModelCheckpoint("weights.best.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')

# fit model to the mean
model, input_lags, preds = build_model_events(1, NUM_LAGS, len(sel)+14, 1)
#print model_weather.summary()
model.fit(
    #[np.concatenate([lags_train, weather_feats_train[:,sel]], axis=1), 
    # np.concatenate([event_feats_train[:,:1]], axis=1)],
    [lags_train[:,:,np.newaxis], lags_event_feats_train[:,:,np.newaxis], np.concatenate([weather_feats_train[:,sel], event_feats_train, lags_event_feats_train], axis=1)],
    y_train,
    batch_size=64,
    epochs=500,
    #validation_split=0.2,
    #validation_data=([np.concatenate([lags_val, weather_feats_val[:,sel]], axis=1), 
    #                  np.concatenate([event_feats_val[:,:1]], axis=1)], y_val),
    validation_data=([lags_val[:,:,np.newaxis], lags_event_feats_val[:,:,np.newaxis], np.concatenate([weather_feats_val[:,sel], event_feats_val, lags_event_feats_val], axis=1)], y_val),
    callbacks=[checkpoint],
    verbose=0)   

print "Total number of iterations:  ", len(model.history.history["loss"])
print "Best loss at iteratation:    ", np.argmin(model.history.history["loss"]), "   Best:", np.min(model.history.history["loss"])
print "Best val_loss at iteratation:", np.argmin(model.history.history["val_loss"]), "   Best:", np.min(model.history.history["val_loss"])

# load weights
model.load_weights("weights.best.hdf5")

print model.evaluate([lags_test[:,:,np.newaxis], lags_event_feats_test[:,:,np.newaxis], np.concatenate([weather_feats_test[:,sel], event_feats_test, lags_event_feats_test], axis=1)], 
                      y_test, verbose=2)

# make predictions
preds_lstm = model.predict([lags_test[:,:,np.newaxis], lags_event_feats_test[:,:,np.newaxis], np.concatenate([weather_feats_test[:,sel], event_feats_test, lags_event_feats_test], axis=1)])
preds_lstm = preds_lstm[:,0] * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lstm)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))


# ---------------------------------------- LSTM with weather + events information (no text) + TEXT

print "\npreparing word embeddings for NNs with text..."

# Build index mapping words in the embeddings set to their embedding vector
embeddings_index = {}
f = open(GLOVE_DIR + 'glove.6B.%dd.txt' % (EMBEDDING_DIM,))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# Vectorize the text samples into a 2D integer tensor and pad sequences
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(event_texts)
sequences_train = tokenizer.texts_to_sequences(event_texts_train)
sequences_val = tokenizer.texts_to_sequences(event_texts_val)
sequences_test = tokenizer.texts_to_sequences(event_texts_test)

word_index = tokenizer.word_index
print 'Found %s unique tokens.' % len(word_index)

data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
data_val = pad_sequences(sequences_val, maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
print 'Shape of train tensor:', data_train.shape
print 'Shape of val tensor:', data_val.shape
print 'Shape of test tensor:', data_test.shape

# Prepare embedding matrix
print('Preparing embedding matrix.')
num_words = min(MAX_NB_WORDS, len(word_index)+1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    #print i
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


def build_model_text(num_inputs, num_lags, num_feat, num_preds):
    input_lags = Input(shape=(num_lags,1))
    input_events = Input(shape=(num_lags,1))
    input_events2 = Input(shape=(num_feat,))
    
    x_lags = Concatenate(axis=2)([input_lags, input_events])
    #x_lags = BatchNormalization()(x_lags)
    x_lags = LSTM(30, 
             kernel_regularizer=keras.regularizers.l2(0.2), 
             recurrent_regularizer=keras.regularizers.l2(0.2), 
             return_sequences=False)(x_lags)
    #x_lags = Activation("relu")(x_lags)
    #x_lags = Flatten()(x_lags)
    x_lags = BatchNormalization()(x_lags)
    #x_lags = Dropout(0.2)(x_lags)
    #x_lags = BatchNormalization()(x_lags)
    
    feat = Concatenate(axis=1)([x_lags, input_events2])
    
    feat = BatchNormalization()(feat)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(50, 3, activation='relu')(embedded_sequences)
    x = MaxPooling1D(3)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(30, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Dropout(0.5)(x)
    x = Conv1D(30, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    #x = Dropout(0.1)(x)
    #x = Conv1D(50, 5, activation='relu')(x)
    #x = MaxPooling1D(5)(x)
    text_embedding = Flatten()(x)
    text_embedding = Dropout(0.5)(text_embedding)
    #text_embedding = Dense(units=100, activation='relu')(text_embedding)
    #text_embedding = Dropout(0.5)(text_embedding)

    feat = Concatenate(axis=1)([feat, text_embedding])

    #preds = Dense(units=num_preds)(feat)
    preds = Dense(units=num_preds, kernel_regularizer=keras.regularizers.l2(0.2))(feat)
    preds = Activation("linear")(preds)
    
    model = Model([input_lags, input_events, input_events2, sequence_input], preds)
    
    rmsp = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    #model.compile(loss="mse", optimizer=rmsp)
    model.compile(loss="mse", optimizer="adam")
    
    return model, input_lags, preds


print "\nrunning LSTM with lags + weather + events + late + text..."

# checkpoint best model
checkpoint = ModelCheckpoint("weights.best.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')

# fit model to the mean
model, input_lags, preds = build_model_text(1, NUM_LAGS, len(sel)+4, 1)
model.fit(
    #[np.concatenate([lags_train, weather_feats_train[:,sel]], axis=1), 
    # np.concatenate([event_feats_train[:,:1]], axis=1)],
    [lags_train[:,:,np.newaxis], lags_event_feats_train[:,:,np.newaxis], 
    np.concatenate([weather_feats_train[:,sel], event_feats_train], axis=1), data_train],
    y_train,
    batch_size=64,
    epochs=500,
    #validation_split=0.2,
    #validation_data=([np.concatenate([lags_val, weather_feats_val[:,sel]], axis=1), 
    #                  np.concatenate([event_feats_val[:,:1]], axis=1)], y_val),
    validation_data=([lags_val[:,:,np.newaxis], lags_event_feats_val[:,:,np.newaxis], 
        np.concatenate([weather_feats_val[:,sel], event_feats_val], axis=1), data_val], y_val),
    callbacks=[checkpoint],
    verbose=0)    

print "Total number of iterations:  ", len(model.history.history["loss"])
print "Best loss at iteratation:    ", np.argmin(model.history.history["loss"]), "   Best:", np.min(model.history.history["loss"])
print "Best val_loss at iteratation:", np.argmin(model.history.history["val_loss"]), "   Best:", np.min(model.history.history["val_loss"])

# load weights
model.load_weights("weights.best.hdf5")

print model.evaluate([lags_test[:,:,np.newaxis], lags_event_feats_test[:,:,np.newaxis], 
                    np.concatenate([weather_feats_test[:,sel], event_feats_test], axis=1), data_test], 
                      y_test, verbose=2)

# make predictions
preds_lstm = model.predict([lags_test[:,:,np.newaxis], lags_event_feats_test[:,:,np.newaxis], 
                        np.concatenate([weather_feats_test[:,sel], event_feats_test], axis=1), data_test])
preds_lstm = preds_lstm[:,0] * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lstm)
print "MAE:  %.3f\tRMSE: %.3f\tR2:   %.3f" % (mae, rmse, r2)
fw_mae.write("%.3f," % (mae,))
fw_rae.write("%.3f," % (rae,))
fw_rmse.write("%.3f," % (rmse,))
fw_rrse.write("%.3f," % (rrse,))
fw_mape.write("%.3f," % (mape,))
fw_r2.write("%.3f," % (r2,))


# ---------------------------------------- LSTM with weather + events information (no text) + event_lags + TEXT


print "\nrunning MLP with lags + weather + events + late + event_lags + text..."

# checkpoint best model
checkpoint = ModelCheckpoint("weights.best.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')

# fit model to the mean
model, input_lags, preds = build_model_text(1, NUM_LAGS, len(sel)+14, 1)
model.fit(
    #[np.concatenate([lags_train, weather_feats_train[:,sel]], axis=1), 
    # np.concatenate([event_feats_train[:,:1]], axis=1)],
    [lags_train[:,:,np.newaxis], lags_event_feats_train[:,:,np.newaxis], 
    np.concatenate([weather_feats_train[:,sel], event_feats_train, lags_event_feats_train], axis=1), data_train],
    y_train,
    batch_size=64,
    epochs=500,
    #validation_split=0.2,
    #validation_data=([np.concatenate([lags_val, weather_feats_val[:,sel]], axis=1), 
    #                  np.concatenate([event_feats_val[:,:1]], axis=1)], y_val),
    validation_data=([lags_val[:,:,np.newaxis], lags_event_feats_val[:,:,np.newaxis], 
        np.concatenate([weather_feats_val[:,sel], event_feats_val, lags_event_feats_val], axis=1), data_val], y_val),
    callbacks=[checkpoint],
    verbose=0)    

print "Total number of iterations:  ", len(model.history.history["loss"])
print "Best loss at iteratation:    ", np.argmin(model.history.history["loss"]), "   Best:", np.min(model.history.history["loss"])
print "Best val_loss at iteratation:", np.argmin(model.history.history["val_loss"]), "   Best:", np.min(model.history.history["val_loss"])

# load weights
model.load_weights("weights.best.hdf5")

print model.evaluate([lags_test[:,:,np.newaxis], lags_event_feats_test[:,:,np.newaxis], 
                    np.concatenate([weather_feats_test[:,sel], event_feats_test, lags_event_feats_test], axis=1), data_test], 
                      y_test, verbose=2)

# make predictions
preds_lstm = model.predict([lags_test[:,:,np.newaxis], lags_event_feats_test[:,:,np.newaxis], 
                        np.concatenate([weather_feats_test[:,sel], event_feats_test, lags_event_feats_test], axis=1), data_test])
preds_lstm = preds_lstm[:,0] * std_test + trend_test
corr, mae, rae, rmse, rrse, mape, r2 = compute_error(y_true, preds_lstm)
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



