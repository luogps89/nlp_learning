# Authoe:LG
# Time:20190829
# Target：solve the Xinhua news agnecy copyed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import jieba

# load data
def load_data():
    data = pd.read_csv('/users/lg/NLP/NLP_Practice/NLP_Practice/data/sqlResult_1558435.csv',encoding='gb18030')
    # print(data.head())
    data = data.dropna(subset=['source', 'content'])
    news_data = pd.DataFrame(data['content'])
    #news_data['y'] = pd.DataFrame(np.where(data['source']=='新华社',1,0))
    news_data['y'] = data.apply(lambda x: 1 if x['source']=='新华社' else 0, axis=1)
    #print(news_data.head())

    text_list = news_data['content'].to_list()
    y = news_data['y'].to_list()
    return news_data,text_list,y

# clean text
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

def tfidf_vect(corpus):
    # use tfidf word2vector
    vect = TfidfVectorizer()
    X = vect.fit_transform(corpus)

    #print(vect.get_feature_names())
    print(X.shape)
    return X

# fit model
def svm_model(X,y):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    # gridsearchcv
    # use svm train data
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svc = SVC(gamma='scale')

    svm_clf = GridSearchCV(svc, parameters, cv=5)
    svm_clf.fit(X_train, y_train)

    print(svm_clf.best_params_)

    y_pred_svm = svm_clf.predict(X_test)
    print(accuracy_score(y_test, y_pred_svm))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def rnd_model(X,y):
    # use rdf
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rnd_clf = RandomForestClassifier(n_estimators=100)
    rnd_clf.fit(X_train, y_train)
    y_pred_rnd = rnd_clf.predict(X_test)
    print(accuracy_score(y_test, y_pred_rnd))

    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred_rnd)  # 输出混淆矩阵
    print(confmat)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()
    y_pred = rnd_clf.predict(X)
    return y_pred

# find plagiarist
def poential_copy_text(news_data,y_pred):
    news_data['y_pred'] = y_pred
    poential_copy_texts = text_list[(text_list.y==0) & (text_list.y_pred==1)]
    print(poential_copy_texts[:5])


if __name__ == '__main__':
    # news_data,text_list ,y = load_data()
    data = pd.read_csv('/users/lg/NLP/NLP_Practice/NLP_Practice/data/sqlResult_1558435.csv',encoding='gb18030')
    # print(data.head())
    data = data.dropna(subset=['source', 'content'])
    news_data = pd.DataFrame(data['content'])
    #news_data['y'] = pd.DataFrame(np.where(data['source']=='新华社',1,0))
    news_data['y'] = data.apply(lambda x: 1 if x['source']=='新华社' else 0, axis=1)
    #print(news_data.head())

    text_list = news_data['content'].to_list()
    y = news_data['y'].to_list()
    corpus = cut_word(text_list)
    X = tfidf_vect(corpus)
    y_pred = rnd_model(X,y)
    poential_copy_text(news_data,y_pred)



'''
思考：
数据思维：是对数据的清洗处理，以及对数据的敏感性，需要获取什么样的数据，能更好的符合业务场景，以及选择什么样的模型来解决
机器学习思维：将一个业务问题，转化为机器学习算法能解决的问题，如本项目，识别出潜在抄袭者
不同的模型，在特定数据集中，会有其表现好的地方

'''