import pandas as pd
import re
from pymorphy2 import MorphAnalyzer
#from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

morph = MorphAnalyzer()
patterns_ru = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
patterns_kk = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
patterns_en = "[А-Яа-я0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
#stopwords_ru = stopwords.words("russian")
stopwords_ru=['другои', 'еи', 'какои', 'мои', 'неи', 'сеичас', 'такои', 'этои','и', 'в', 'во', 'не', 'что', 'он','на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне','было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между']
stopwords_en=['i','me', 'my', 'myself', 'we', 'our', 'ours','ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any','both', 'each', 'few', 'more', 'most', 'other','some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stopwords_kk=['ах', 'ох', 'эх', 'ай', 'эй', 'ой', 'тағы', 'тағыда', 'әрине', 'жоқ', 'сондай', 'осындай', 'осылай', 'солай', 'мұндай', 'бұндай', 'мен', 'сен', 'ол', 'біз', 'біздер', 'олар', 'сіз', 'сіздер', 'маған', 'оған', 'саған', 'біздің', 'сіздің', 'оның', 'бізге', 'сізге','оларға', 'біздерге', 'сіздерге', 'оларға', 'менімен', 'сенімен', 'онымен', 'бізбен', 'сізбен', 'олармен', 'біздермен', 'сіздермен', 'менің', 'сенің', 'біздің', 'сіздің', 'оның', 'біздердің', 'сіздердің', 'олардың', 'маған', 'саған', 'оған', 'менен', 'сенен', 'одан', 'бізден', 'сізден', 'олардан', 'біздерден', 'сіздерден', 'олардан', 'айтпақшы', 'сонымен', 'сондықтан', 'бұл', 'осы', 'сол', 'анау', 'мынау', 'сонау', 'осынау', 'ана', 'мына', 'сона', 'әні', 'міне', 'өй', 'үйт', 'бүйт', 'біреу', 'кейбіреу', 'кейбір', 'қайсыбір', 'әрбір', 'бірнеше', 'бірдеме', 'бірнеше', 'әркім', 'әрне', 'әрқайсы', 'әрқалай', 'әлдекім', 'әлдене', 'әлдеқайдан', 'әлденеше', 'әлдеқалай', 'әлдеқашан', 'алдақашан', 'еш', 'ешкім', 'ешбір', 'ештеме', 'дәнеңе', 'ешқашан', 'ешқандай', 'ешқайсы', 'емес', 'бәрі', 'барлық', 'барша', 'бар', 'күллі', 'бүкіл', 'түгел', 'өз', 'өзім', 'өзің', 'өзінің', 'өзіме', 'өзіне', 'өзімнің', 'өзі', 'өзге', 'менде', 'сенде', 'онда', 'менен', 'сенен\tонан', 'одан', 'ау', 'па', 'ей', 'әй', 'е', 'уа', 'уау', 'уай', 'я', 'пай', 'ә', 'о', 'оһо', 'ой', 'ие', 'аһа', 'ау', 'беу', 'мәссаған', 'бәрекелді', 'әттегенай', 'жаракімалла', 'масқарай', 'астапыралла', 'япырмай', 'ойпырмай', 'кәне', 'кәнеки', 'ал', 'әйда', 'кәні', 'міне', 'әні', 'сорап', 'қош-қош', 'пфша', 'пішә', 'құрау-құрау', 'шәйт', 'шек', 'моһ', 'тәк', 'құрау', 'құр', 'кә', 'кәһ', 'күшім', 'күшім', 'мышы', 'пырс', 'әукім', 'алақай', 'паһ-паһ', 'бәрекелді', 'ура', 'әттең', 'әттеген-ай', 'қап', 'түге', 'пішту', 'шіркін', 'алатау', 'пай-пай', 'үшін', 'сайын', 'сияқты', 'туралы', 'арқылы', 'бойы', 'бойымен', 'шамалы', 'шақты', 'қаралы', 'ғұрлы', 'ғұрлым', 'шейін', 'дейін', 'қарай', 'таман', 'салым', 'тарта', 'жуық', 'таяу', 'гөрі', 'бері', 'кейін', 'соң', 'бұрын', 'бетер', 'қатар', 'бірге', 'қоса', 'арс', 'гүрс', 'дүрс', 'қорс', 'тарс', 'тырс', 'ырс', 'барқ', 'борт', 'күрт', 'кірт', 'морт', 'сарт', 'шырт', 'дүңк', 'күңк', 'қыңқ', 'мыңқ', 'маңқ', 'саңқ', 'шаңқ', 'шіңк', 'сыңқ', 'таңқ', 'тыңқ', 'ыңқ', 'болп','былп', 'жалп', 'желп', 'қолп', 'ірк', 'ырқ', 'сарт-сұрт', 'тарс-тұрс', 'арс-ұрс', 'жалт-жалт', 'жалт-жұлт', 'қалт-қалт', 'қалт-құлт', 'қаңқ-қаңқ', 'қаңқ-құңқ', 'шаңқ-шаңқ', 'шаңқ-шұңқ', 'арбаң-арбаң', 'бүгжең-бүгжең', 'арсалаң-арсалаң', 'ербелең-ербелең', 'батыр-бұтыр', 'далаң-далаң','тарбаң-тарбаң', 'қызараң-қызараң', 'қаңғыр-күңгір', 'қайқаң-құйқаң', 'митың-митың', 'салаң-сұлаң', 'ыржың-тыржың', 'бірақ', 'алайда', 'дегенмен', 'әйтпесе', 'әйткенмен', 'себебі', 'өйткені', 'сондықтан', 'үшін', 'сайын', 'сияқты', 'туралы', 'арқылы', 'бойы', 'бойымен', 'шамалы', 'шақты', 'қаралы', 'ғұрлы', 'ғұрлым', 'гөрі', 'бері', 'кейін', 'соң', 'бұрын', 'бетер', 'қатар', 'бірге', 'қоса', 'шейін', 'дейін', 'қарай', 'таман', 'салым', 'тарта', 'жуық', 'таяу', 'арнайы', 'осындай', 'ғана', 'қана', 'тек', 'әншейін']

def lemmatize_en(doc):
    doc = re.sub(patterns_en, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token.lower() not in stopwords_en:
            token = token.strip()
            #token = morph.normal_forms(token)[0]
            
            tokens.append(token.lower())
    if len(tokens) > 0:
        return tokens
    return None

def lemmatize_ru(doc):
    doc = re.sub(patterns_ru, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token.lower() not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            
            tokens.append(token.lower())
    if len(tokens) > 0:
        return tokens
    return None

def lemmatize_kk(doc):
    doc = re.sub(patterns_kk, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token.lower() not in stopwords_kk:
            token = token.strip()
            #token = morph.normal_forms(token)[0]
            
            tokens.append(token.lower())
    if len(tokens) > 0:
        return tokens
    return None

df_ru=pd.read_excel('catalog_ru.xlsx',sheet_name='1')
df_kk=pd.read_excel('catalog_kk.xlsx',sheet_name='1')
df_en=pd.read_excel('catalog_en.xlsx',sheet_name='1')

df2_ru=pd.read_excel('catalog2_ru.xlsx',sheet_name='1')
df2_kk=pd.read_excel('catalog2_kk.xlsx',sheet_name='1')
df2_en=pd.read_excel('catalog2_en.xlsx',sheet_name='1')

data_ru=df_ru['NAME'].apply(lemmatize_ru).str.join(' ').tolist()
data_kk=df_kk['NAME'].apply(lemmatize_kk).str.join(' ').tolist()
data_en=df_en['NAME'].apply(lemmatize_en).str.join(' ').tolist()

data2_ru=df2_ru['NAME'].apply(lemmatize_ru).str.join(' ').tolist()
data2_kk=df2_kk['NAME'].apply(lemmatize_kk).str.join(' ').tolist()
data2_en=df2_en['NAME'].apply(lemmatize_en).str.join(' ').tolist()

v = TfidfVectorizer(input='content',
                    encoding='utf-8', decode_error='replace', strip_accents='unicode',
                    lowercase=True, analyzer='word', stop_words=stopwords_kk,
                    #token_pattern=r'(?u)\b[а-яА-Я_][а-яА-Я0-9_]+\b',
                    #token_pattern=patterns_kk,
                    #ngram_range=(1, 2),
                    ngram_range=(1, 1),
                    max_features=20000,
                    norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                    max_df=0.1, min_df=5
                    #max_df=1, min_df=1
                   )

v2 = TfidfVectorizer(input='content',
                    encoding='utf-8', decode_error='replace', strip_accents='unicode',
                    lowercase=True, analyzer='word', stop_words=stopwords_kk,
                    #token_pattern=r'(?u)\b[а-яА-Я_][а-яА-Я0-9_]+\b',
                    #token_pattern=patterns_kk,
                    #ngram_range=(1, 2),
                    ngram_range=(1, 1),
                    max_features=20000,
                    norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True,
                    max_df=0.1, min_df=5
                    #max_df=1, min_df=1
                   )


import pickle
data_kk=[x if x!=None else '' for x in data_kk]
tfidf_vectorizer_vectors=v.fit_transform(data_kk)
data_temp_kk=pd.DataFrame(tfidf_vectorizer_vectors.T.todense(),index=v.get_feature_names())#[1].sort_values(ascending=False).head(30)

pickle.dump(data_temp_kk,open('catalog_vec_tfidf_kk.pkl','wb'))
pickle.dump(v,open('model_tfidf_kk.pkl','wb'))
pickle.dump(df_kk,open('catalog_kk.pkl','wb'))

data2_kk=[x if x!=None else '' for x in data2_kk]
tfidf_vectorizer_vectors2=v2.fit_transform(data2_kk)
data2_temp_kk=pd.DataFrame(tfidf_vectorizer_vectors2.T.todense(),index=v2.get_feature_names())#[1].sort_values(ascending=False).head(30)

pickle.dump(data2_temp_kk,open('catalog2_vec_tfidf_kk.pkl','wb'))
pickle.dump(v2,open('model2_tfidf_kk.pkl','wb'))
pickle.dump(df2_kk,open('catalog2_kk.pkl','wb'))




pickle.dump(df_en,open('catalog_en.pkl','wb'))
pickle.dump(df_ru,open('catalog_ru.pkl','wb'))

pickle.dump(df2_en,open('catalog2_en.pkl','wb'))
pickle.dump(df2_ru,open('catalog2_ru.pkl','wb'))



from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np
import os
import pandas as pd

model_transformer=SentenceTransformer(os.getcwd()+'/'+'sentence-transformers_paraphrase-multilingual-mpnet-base-v2')

data_temp_en=model_transformer.encode(data_en)
data_temp_ru=model_transformer.encode(data_ru)

data2_temp_en=model_transformer.encode(data2_en)
data2_temp_ru=model_transformer.encode(data2_ru)


pickle.dump(data_temp_en,open('data_temp_en.pkl','wb'))
pickle.dump(data_temp_ru,open('data_temp_ru.pkl','wb'))

pickle.dump(data2_temp_en,open('data2_temp_en.pkl','wb'))
pickle.dump(data2_temp_ru,open('data2_temp_ru.pkl','wb'))

pickle.dump(model_transformer,open('model_transformer.pkl','wb'))
