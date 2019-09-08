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

temp = []
counter = 0

def storage(splitter):
    global temp
    global counter
    temp = splitter
    counter += 1

def give():
    if counter > 1:
        return
    elif counter == 1:
        return temp

logging.basicConfig(level=logging.DEBUG)

if 'win32' in sys.platform:
    # Windows specific event-loop policy
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

async def handler(bot, event):
    channel = event.msg.channel
    sentences = None
    if event.msg.content.type_name != chat1.MessageTypeStrings.ATTACHMENT.value and event.msg.content.type_name != chat1.MessageTypeStrings.TEXT.value:
        return

    elif event.msg.content.type_name == chat1.MessageTypeStrings.ATTACHMENT.value:
        id = event.msg.id
        print(id)
        await bot.chat.download(channel, id, "test.txt")
        with open("test.txt") as f:
            a = f.read()
            splitter = a.split(".")
        c = TOSmain(a)
        await bot.chat.send(channel, str(c))
        storage(splitter)


    elif len(event.msg.content.text.body) < 40:
        wok = event.msg.content.text.body
        inputter = wok
        splitter = give()
        if sentencefind(inputter, splitter) == None:
            pass
        else:
            sentences = splitter[sentencefind(inputter, splitter)]

        await bot.chat.send(channel, sentences)




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
