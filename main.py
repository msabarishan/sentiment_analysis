from typing import Optional
from fastapi import FastAPI, HTTPException

import pandas as pd
import numpy as np
import datetime as dt
from sqlalchemy import create_engine, select, MetaData, Table, and_

#Importing Dataset


import json

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

user = 'postgres'
password = 'EywaAnalytics1!'
host = 'db.evxjorqjydsbdrshnocw.supabase.co'
port = 6543
database = 'postgres'

engine = create_engine(
        url="postgresql://{0}:{1}@{2}:{3}/{4}".format(
            user, password, host, port, database
        )
    )

sql1='Select * from review_topic_teach;'
connection = engine.connect()
results = connection.execute(sql1).fetchall()
connection.close()
df=pd.DataFrame(results)

sql2='Select * from gmb_rev;'
connection = engine.connect()
results = connection.execute(sql2).fetchall()
connection.close()
test_case=pd.DataFrame(results)

sql3='Select * from review_sentiment_teach;'
connection = engine.connect()
results = connection.execute(sql3).fetchall()
df1=pd.DataFrame(results)
connection.close()

comments_df = df[['comment','Topic']]

comments_df["Topic"].value_counts()
sentiment_label = comments_df.Topic.factorize()

comments = comments_df.comment.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(comments)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(comments)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  

history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    #print("Predicted label: ", sentiment_label[1][prediction])
    return sentiment_label[1][prediction]


test_case["category"]=""

#j=len(test_case.index)
j=50
i=0
for i in range(0,j):
    a=test_case["comment"][i]
    test_case.at[i,'category']=predict_sentiment(a)
    i=i+1


comments_df = df1[['comment','sentiment']]

comments_df["sentiment"].value_counts()
sentiment_label = comments_df.sentiment.factorize()

comments = comments_df.comment.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(comments)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(comments)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)

embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  

history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(model.predict(tw).round().item())
    #print("Predicted label: ", sentiment_label[1][prediction])
    return sentiment_label[1][prediction]
test_case_fil=test_case[test_case['category'] == 'sales']
pd.options.mode.chained_assignment = None
test_case_fil = test_case_fil.reset_index()
test_case_fil["sentiment"]=""
#j=len(test_case_fil.index)
j=30
i=0
for i in range(0,j):
    b=test_case["comment"][i]
    test_case_fil.at[i,'sentiment']=predict_sentiment(b)
    i=i+1      
        
positive=test_case_fil['sentiment'].value_counts()['positive']
negative=test_case_fil['sentiment'].value_counts()['negative']


result = test_case_fil.to_json(orient="index")
parsed = json.loads(result)

rating_counts=test_case_fil['rating'].value_counts()
rating_result = rating_counts.to_json(orient="index")
rating_parsed = json.loads(rating_result)


rating_avg=test_case_fil['rating'].mean()

rating_nos=len(test_case_fil.index)

df=test_case_fil
df['comment']=df['comment'].str.lower()
df['comment']=df['comment'].str.replace("(::).*/","")
df['comment']=df['comment'].str.replace("("," ")
df['comment']=df['comment'].str.replace(")"," ")
df['comment']=df['comment'].str.replace(","," ")
df['comment']=df['comment'].str.replace("&"," ")
df['comment']=df['comment'].str.replace("/"," ")
df['comment']=df['comment'].str.replace("."," ")
df['comment']=df['comment'].str.replace("\n"," ")
word_count=df.comment.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
word_count_json=word_count.to_json(orient="index")
word_count_parsed = json.loads(word_count_json)

test_case_fil["created_at"] = test_case_fil["created_at"].astype('datetime64[ns]')
transaction_df= test_case_fil[['sentiment','created_at']]

def get_month(x): return dt.datetime(x.year, x.month, 1) 

# Create transaction_date column based on month and store in TransactionMonth
transaction_df['TransactionMonth'] = transaction_df['created_at'].apply(get_month)

transaction_df['year_month'] = transaction_df['TransactionMonth'].dt.strftime('%Y-%m')
transaction_df_new=transaction_df[['sentiment','year_month']]
transaction_df_new=transaction_df_new.groupby(['year_month','sentiment']).size().reset_index(name='counts')
positive_chart=transaction_df_new[transaction_df_new['sentiment'] == 'positive']
negative_chart=transaction_df_new[transaction_df_new['sentiment'] == 'negative']
positive_chart=positive_chart.drop(['sentiment'], axis=1)
negative_chart=negative_chart.drop(['sentiment'], axis=1)
result1 = positive_chart.to_json(orient="index")
parsed1 = json.loads(result1)    
result2 = negative_chart.to_json(orient="index")
parsed2 = json.loads(result2)                                         

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "New_World"}

@app.get("/sentiment")
def read_root():
  return parsed

@app.get("/positive")
def read_root():
  return {"Value": f"{positive}"}

@app.get("/negative")
def read_root():
  return {"Value": f"{negative}"}

@app.get("/word_cloud")
def read_root():
  return word_count_parsed

@app.get("/positive-chart")
def read_root():
  return parsed1

@app.get("/negative-chart")
def read_root():
  return parsed2

@app.get("/ratings")
def read_root():
  return rating_parsed

@app.get("/avg_ratings")
def read_root():
  return {"avg_rating": f"{rating_avg}"}

@app.get("/ratings_nos")
def read_root():
  return {"review_nos": f"{rating_nos}"}