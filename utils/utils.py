import os
import pickle
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import spacy
import json
from icecream import ic
import re
import os

def create_user_project_dir(user,project):
    path=os.path.join('training_data/' + user)
    #ic(path)
    if not os.path.isdir(path):
        ic()
        os.mkdir(path)
    path = os.path.join(path,project)
    #ic(path)
    if not os.path.isdir(path):
        os.mkdir(path)
    #ic(path)
    return 'success'

def extractFromUserData(data):
    module_data={}
    for entry in data:
        key=entry['lName']
        value=entry['lData']
        if key not in module_data.keys():
            module_data[key]=list([value])
        else:
            module_data[key].append(value)
    return module_data

def data_preprocessing_train(data_dict):
    nlp=spacy.load('en_core_web_sm')
    pattern = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    df = pd.DataFrame(columns=['target', 'text'])
    for key in data_dict.keys():
        clean_text=[]
        for line in data_dict[key]:
            doc=nlp(line)
            for token in doc:
                clean = re.sub(pattern, '', str(token.lemma_).lower())
                if clean not in string.punctuation:
                    clean_text.append(clean)
        df=df.append({'target':key,'text':clean_text},ignore_index=True)
    return df



def processing_data_train(json_path):
    with open(json_path,'r') as f:
        data_dict=json.load(f)

    ##data cleaning
    clean_df = data_preprocessing_train(data_dict)
    # converting preprocesed data from list to string to use in tfIdf
    clean_df['text'] = [" ".join(value) for value in clean_df['text'].values]

    return clean_df


def trainingModel(json_path,model_path):
    data_df=processing_data_train(json_path)

    tf=TfidfVectorizer(max_df=0.95,stop_words='english')

    tf.fit(data_df['text'])

    with open(model_path+'/vectorizer.pickle','wb') as f:
        pickle.dump(tf,f)


    x=data_df['text']
    y=data_df['target']

    x_vector=tf.transform(x)

    model=MultinomialNB()

    model.fit(x_vector,y)

    with open(model_path+'/modelforpredict.sav','wb') as f:
        pickle.dump(model,f)

    return 'success_trainingModel'



