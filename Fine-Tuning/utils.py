import pandas as pd
import os
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def label_class(sentiment):
    if sentiment=="Negative":
        return 0
    elif sentiment=="Neutral":
        return 1
    elif sentiment=="Positive":
        return 2
    elif sentiment=="Irrelevant":
        return 3


def loading_data(training_path: str,validation_path: str):
    try:
        columns = ["id", "game", "sentiment", "text"]
        training=pd.read_csv(training_path,header=None,names=columns)
        training.drop(["id","game"],axis=1,inplace=True)
        training.dropna(inplace=True)
        training['sentiment']=training['sentiment'].map(label_class)
        validation=pd.read_csv(validation_path,header=None,names=columns)
        validation.drop(["id","game"],axis=1,inplace=True)
        validation.dropna(inplace=True)
        validation['sentiment']=validation['sentiment'].map(label_class)
        return training, validation
    except FileNotFoundError:
        print("train.csv not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def text_cleaning(text):
    text['text'] = text['text'].str.lower()  # convert to lowercase
    text['text'] = text['text'].apply(lambda x: re.sub(re.compile(r'http\S+|https\S+'), "", x))  # remove url 
    text['text'] = text['text'].apply(lambda x: re.sub(re.compile('<.*?>'), "", x))   # remove html tags
    text['text'] = text['text'].apply(lambda x: re.sub(re.compile("[^A-Za-z]")," ", x))  # remove non-alphanomeric characters
    text['text']= text['text'].apply(lambda x: re.sub(re.compile(' +'),' ', x))  # remove extra spaces 
    return text


def concatenation(set1,set2):
    full=pd.concat([set1,set2],ignore_index=True)
    #shuffle
    full_df = shuffle(full, random_state=42).reset_index(drop=True)
    #splitting
    train_data, test_data = train_test_split(
    full_df,
    test_size=0.25,
    random_state=42,
    stratify=full_df["sentiment"]
)
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    return train_data,test_data






