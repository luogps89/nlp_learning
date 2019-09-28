# using kmeans to make a news cluster
# author:lg
# time:20190910
import pandas as pd
import numpy as np
import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pickle

# extract type
def convert(data):
    global type_dict
    global type_count
    type = json.loads(data['feature'])['type']
    if not type in type_dict:
        type_dict[type] = type_count
        type_count += 1
    return type_dict[type]

# clean data
def clean_text(data):
    data = data.replace('\r','')
    data = data.replace('\n','')
    data = data.replace(' ','')
    data = data.replace('\u3000', '')
    data.replace('/', '')
    return data

# cut words
def cut_word(text_list):
    corpus = []
    for text in text_list:
        if type(text) == float:
            corpus.append('null')
        else:
            text = clean_text(text)
            text = jieba.cut(text)
            line = ' '.join(text)
            corpus.append(line)
    return corpus


if __name__ == '__main__':
    # load data
    print('开始执行')
    data = pd.read_csv(r'/home/student/sqlResult_1558435.csv', encoding='gb18030',usecols=['content', 'feature'])
    print('已完成数据加载')
    data = data.fillna('')

    type_dict = dict()
    type_count = 0
    data['target'] = data.apply(convert, axis=1)
    news_content = clean_text(data['content'])
    corpus = cut_word(news_content.to_list())
    print('已完成切词')

    # 文本向量化
    vect = TfidfVectorizer()
    X = vect.fit_transform(corpus)
    print('已完成文本向量化')

    # kmeans 聚类
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=5)
    print('开始训练kmeans模型')
    km.fit(X)
    print('模型训练完成')
    with open('/home/student/lg/save_model/clf.pickle', 'wb') as f:
        pickle.dump(km, f)

    print('模型保存成功')
