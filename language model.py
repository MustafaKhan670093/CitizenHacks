#!/usr/bin/env python3

###################################
# WHAT IS IN THIS EXAMPLE?
#
# This bot listens in one channel and reacts to every text message.
###################################
import boto3
import time
import asyncio
import logging
import os
import sys
import pykeybasebot.types.chat1 as chat1
from pykeybasebot import Bot
from pykeybasebot import chat_client
import numpy as np
import pandas as pd
import requests
from datetime import datetime
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from yahoo_fin import stock_info as si
from cryptowatch_client import Client
from datetime import date
client = boto3.client('comprehend')

def stock_pred(ticker):

    config = tf.ConfigProto( device_count = {''
                                             'GPU': 1 , 'CPU': 6} )
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)



    scaler = MinMaxScaler(feature_range=(0, 1))

    today = date.today()
    end =  str(today.month) + "/" + str(today.day) + "/" + str(today.year)
    print(end)
    start = str(today.month - 4) + "/" + str(today.day) + "/" + str(today.year - 1)
    print(start)


    time = '1'


    try:
        si.get_live_price(ticker)
    except BaseException:
        raise

    week = si.get_data(ticker, start_date=start, end_date=end)

    week = week.iloc[:, 0]
    week.to_numpy()
    stock_price = []

    for i in range(0, week.shape[0]):
        stock_price.append(week[i])
    stock_price = [stock_price]
    stock_price = np.asarray(stock_price, dtype=np.float32)
    stock_price = np.reshape(stock_price, (stock_price.shape[1], stock_price.shape[0]))
    training_processed = stock_price
    training = training_processed
    testing = training_processed
    training_scaled = scaler.fit_transform(training)
    testing_scaled = scaler.fit_transform(testing)

    features_set = []
    labels = []

    for i in range(len(training_scaled)):
        features_set.append(training_scaled[i])
    features_set.remove(features_set[i])

    for i in range(1, len(training_scaled)):
        labels.append(training_scaled[i])

    features_set, labels = np.array(features_set), np.array(labels)

    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True, activation= "relu"))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True, activation= "relu"))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True, activation= "relu"))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True, activation= "relu"))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(features_set, labels, epochs=25 , batch_size=64)

    a = []
    counter = 0
    a.append(np.reshape(training_scaled[training_scaled.size - 1], (1, 1, 1)))
    while counter < int(time):
        a.append(model.predict(np.reshape(a[len(a) - 1], (1, 1, 1))))
        counter += 1



    a = np.reshape(a, (len(a), 1))

    temp = np.reshape(testing_scaled, (testing_scaled.size, 1, 1))
    temp = model.predict(temp)
    temp = np.reshape(temp, (testing_scaled.size, 1))
    temp = scaler.inverse_transform(temp)
    a = scaler.inverse_transform(a)

    a = np.append(temp,a)
    a = a.tolist()
    training_scaled = scaler.inverse_transform(training_scaled)
    training_scaled = training_scaled.tolist()


    training_final = []
    for i in range(len(training_scaled)):
        training_final.append(training_scaled[i][0])

    plt.figure(figsize=(10, 6))
    plt.plot(training_final, color='blue', label='Actual ' + ticker + ' Stock Price')
    plt.plot(a, color='red', label='Predicted ' + ticker + ' Stock Value')
    plt.xlabel('Date (Days)')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('temp.png')

    return training_final


def crypto_predict(ticker):

    config = tf.ConfigProto( device_count = {''
                                             'GPU': 1 , 'CPU': 6} )
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)



    scaler = MinMaxScaler(feature_range=(0, 1))



    client = Client()

    periods = '60'
    resp = requests.get(ticker, params={
        'periods': periods
    })
    data = resp.json()
    df = pd.DataFrame(data['result'][periods], columns=[
        'CloseTime', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice', 'Volume', 'NA'
    ])
    df.drop(columns=['NA'], inplace=True)
    df['CloseTime'] = pd.to_datetime(df['CloseTime'], unit='s')
    prices = df.loc[:, 'ClosePrice']
    week = prices
    week.to_numpy()
    stock_price = []

    for i in range(0, week.shape[0]):
        stock_price.append(week[i])
    stock_price = [stock_price]
    stock_price = np.asarray(stock_price, dtype=np.float32)
    stock_price = np.reshape(stock_price, (stock_price.shape[1], stock_price.shape[0]))
    training_processed = stock_price
    training = training_processed
    testing = training_processed
    training_scaled = scaler.fit_transform(training)
    testing_scaled = scaler.fit_transform(testing)

    features_set = []
    labels = []

    for i in range(len(training_scaled)):
        features_set.append(training_scaled[i])
    features_set.remove(features_set[i])

    for i in range(1, len(training_scaled)):
        labels.append(training_scaled[i])

    features_set, labels = np.array(features_set), np.array(labels)

    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True, activation= "relu"))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True, activation= "relu"))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True, activation= "relu"))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=True, activation= "relu"))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(features_set, labels, epochs=30 , batch_size=64)

    a = []
    counter = 0
    a.append(np.reshape(training_scaled[training_scaled.size - 1], (1, 1, 1)))
    while counter < int(1):
        a.append(model.predict(np.reshape(a[len(a) - 1], (1, 1, 1))))
        counter += 1



    a = np.reshape(a, (len(a), 1))

    temp = np.reshape(testing_scaled, (testing_scaled.size, 1, 1))
    temp = model.predict(temp)
    temp = np.reshape(temp, (testing_scaled.size, 1))
    temp = scaler.inverse_transform(temp)
    a = scaler.inverse_transform(a)

    a = np.append(temp,a)
    a = a.tolist()
    training_scaled = scaler.inverse_transform(training_scaled)
    training_scaled = training_scaled.tolist()


    training_final = []
    for i in range(len(training_scaled)):
        training_final.append(training_scaled[i][0])

    plt.figure(figsize=(10, 6))
    plt.plot(training_final, color='blue', label='Actual ' + ticker + ' Stock Price')
    plt.plot(a, color='red', label='Predicted ' + ticker + ' Crypto Value')
    plt.xlabel('Date (Mins)')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('temp.png')

    return training_final


def sentencefind(string, list):
    for i in list:
        if string in i:
            return list.index(i)


def languagedet(text):
    responselang = client.batch_detect_dominant_language(
        TextList=[
            text,
        ]
    )
    return responselang

def TOSmain(string):

    def TOS():
        langdict = languagedet(string)['ResultList'][0]['Languages']
        for i in langdict:
            language = i['LanguageCode']
            break
        response = client.batch_detect_key_phrases(
            TextList=[
                string,
            ],
            LanguageCode=language
        )
        lister = []
        b = response['ResultList'][0]['KeyPhrases']
        for i in b:
            n = i['Text']
            if "Terms" in n:
                continue
            elif n in lister:
                continue
            else:
                lister.append(n)



        return lister
    x = TOS()
    return x



logging.basicConfig(level=logging.DEBUG)

if 'win32' in sys.platform:
    # Windows specific event-loop policy
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())



async def handler(bot, event):
    channel = event.msg.channel
    if event.msg.content.type_name != chat1.MessageTypeStrings.TEXT.value:
        return
    elif event.msg.content.type_name == chat1.MessageTypeStrings.TEXT.value:
        a = event.msg.content.text.body
        if len(a) > 6:
        #await bot.chat.send(channel, "Please wait a bit! ^_^")
            b = crypto_predict(a)
            await bot.chat.send(channel, "Right now it's " + str(b[len(b)-2]) + ". " + "The price in one minute is going to be: " + str(b[len(b)-1]))
            await bot.chat.attach(channel, r"C:\Users\adity\OneDrive\Desktop\PrivacyBin\temp.png", "Predicted vs Actual Stock price")
        elif len(a) < 6:
        #await bot.chat.send(channel, "Please wait a bit! ^_^")
            b = stock_pred(a)
            await bot.chat.send(channel, "Right now it's " + str(b[len(b)-2]) + ". " + "The price in one day is going to be: " + str(b[len(b)-1]) + " ")
            await bot.chat.attach(channel, r"C:\Users\adity\OneDrive\Desktop\PrivacyBin\temp.png", "Predicted vs Actual Stock price")



listen_options = {
    "local": True,
    "wallet": True,
    "dev": True,
    "hide-exploding": False,
    "filter_channel": None,
    "filter_channels": None,

}

bot = Bot(username="citizenbot", paperkey= "identify virtual arctic hotel recall already neutral card cabin promote federal romance lend", handler=handler)

asyncio.run(bot.start(listen_options))


