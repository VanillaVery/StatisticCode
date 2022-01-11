import numpy as np
from konlpy.tag import Okt
import tensorflow as tf
from tensorflow.keras import layers
import re
import psycopg2
import os
import itertools
import json
import random
import pandas as pd
from tqdm import tqdm


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#
#
# conn_string = "host=maderi.cka0sue4nsid.ap-northeast-2.rds.amazonaws.com dbname = maderi_raw user = dmk_datasci_yjy password = yjy0408** port=5432"
# conn = psycopg2.connect(conn_string)
# cur = conn.cursor()
#
# query="""
# SELECT content FROM t_buzz_contents_20211108
# """
#
# testdata = pd.read_sql_query(query, con=conn)

jytest=pd.read_csv("C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/바탕 화면/lime/jytest.csv")

jytrain=pd.read_csv("C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/바탕 화면/lime/jytrain.csv")

#한글만 남기고 제거
jytrain['document'] = jytrain['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
jytest['document'] = jytest['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

#토큰화
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
#train_data에 형태소 분석기를 사용하여 토큰화를 하면서 불용어를 제거하여 X_train에 저장
# len1=[]
# for sentence in jytrain['document']:
#     len1.append(len(sentence.split(' ')))
# pd.DataFrame(len1).sort_values(by=0).hist()
okt = Okt()

outindex_test=[]
for i,sentence in enumerate(jytest['document']):
    if len(sentence.split(' '))>=100:
        outindex_test.append(i)


jytest=jytest.drop(outindex_test)

outindex_train=[]
for i,sentence in enumerate(jytrain['document']):
    if len(sentence.split(' '))>=100:
        outindex_train.append(i)


jytrain=jytrain.drop(outindex_train)


X_train = []

for sentence in tqdm(jytrain['document'][:2220]):

    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_train.append(stopwords_removed_sentence)


X_test = []
for sentence in tqdm(jytest['document']):
    tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
    stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
    X_test.append(stopwords_removed_sentence)





import sklearn.feature_extraction
import sklearn.metrics
from sklearn.naive_bayes import MultinomialNB

#TF-IDF를 사용해서 문서를 숫자 벡터로 변환하는 전처리 과정
vectorizer=sklearn.feature_extraction.text.TfidfVectorizer(preprocessor=' '.join,lowercase=False)
train_vectors=vectorizer.fit_transform(X_train)
test_vectors=vectorizer.transform(X_test)

#학습하기
nb=MultinomialNB(alpha=.01)
nb.fit(train_vectors,jytrain.iloc[:2220,2])

#테스트하기
pred=nb.predict(test_vectors)
sklearn.metrics.f1_score(jytest.iloc[:,2],pred,average='weighted')

#파이프라인 기술을 사용해 테스트 데이터 인덱스 0번에 데이터 벡터라이저와 카테고리 분류를 한꺼번에 수행
from sklearn.pipeline import make_pipeline

pipe=make_pipeline(vectorizer,nb)
predict_classes=pipe.predict_proba([X_test[44]]).round(3)[0]
predict_classes

#LIME 텍스트 설명체를 선언하는 코드
from lime.lime_text import LimeTextExplainer
class_names={0:'거래판매',1:'렌탈',2:'부동산',3:'수리',4:'이벤트',5:'인사글',6:'종교',7:'주식',8:'체험단'}
explainer=LimeTextExplainer(class_names=class_names)

exp=explainer.explain_instance(jytest['document'].iloc[44],pipe.predict_proba,top_labels=2)
exp.available_labels()
exp.save_to_file("C:/Users/윤유진/OneDrive - 데이터마케팅코리아 (Datamarketingkorea)/바탕 화면/lime/test44.html")

jytest.iloc[44,:]