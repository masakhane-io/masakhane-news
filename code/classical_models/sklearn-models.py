import os
import pandas as pd
import tqdm
import numpy as np

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import os

folder_name = '../../data/'

feature_column = "headline"
label_column = "category"

np.random.seed(42)

language_list = ['amh','eng','fra','hau','ibo','lin','pcm','run','swa','yor']
print(language_list)
for language in language_list:
    print('-------------------------------------------------')
    print(f'--------------Working on {language}-----------------')
    
    train_data = pd.read_csv(f'{folder_name}/{language}/train.tsv',sep='\t')
    dev_data = pd.read_csv(f'{folder_name}/{language}/dev.tsv',sep='\t')
    test_data = pd.read_csv(f'{folder_name}/{language}/dev.tsv',sep='\t')

    print(f' Training set size : {train_data.size}   Dev set size: {dev_data.size}')
    
    all_text_list  = train_data[feature_column].values.tolist()+dev_data[feature_column].values.tolist() 
    
    print('[INFO] Sample data \n',all_text_list[:3])
    
    train_text,train_label = train_data[feature_column].values.tolist(),train_data[label_column].values.tolist()
    dev_text,dev_label = dev_data[feature_column].values.tolist(),dev_data[label_column].values.tolist()
    test_text,test_label = test_data[feature_column].values.tolist(),test_data[label_column].values.tolist()

    
    unique_label = train_data[label_column].unique().tolist()
    
    print('[INFO] Found Labels : ',unique_label)
    # CountVectorizer
    vectorizer = CountVectorizer(analyzer='char_wb',ngram_range=(1, 3))
    vectorizer.fit_transform(all_text_list)
    
    X_train = vectorizer.transform(train_text).toarray()
    X_dev= vectorizer.transform(dev_text).toarray()
    X_test= vectorizer.transform(test_text).toarray()
    
    y_train = []
    for i in train_label:
        y_train.append(unique_label.index(i))

    y_dev = []
    for i in dev_label:
        y_dev.append(unique_label.index(i))
        
    y_test = []
    for i in test_label:
        y_test.append(unique_label.index(i))
        
    print(f'Sizes : {X_train.shape,X_dev.shape,X_test.shape,len(y_train),len(y_dev),len(y_test)}')


    
    print('=======   GaussianNB   =========')

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predict Class
    y_pred = classifier.predict(X_dev)

    # Accuracy 
    accuracy = metrics.accuracy_score(y_dev, y_pred)
    f1 = metrics.f1_score(y_dev, y_pred, average='macro')


    print(f'acc: {accuracy}     |  f1_score: {f1}')
    print(metrics.classification_report(y_dev, y_pred, target_names=unique_label))
    
    
    if not os.path.exists(f"{language}/GaussianNB"):
        os.makedirs(f"{language}/GaussianNB")
        
    acc = metrics.accuracy_score(y_dev, y_pred)
    f1 = metrics.f1_score(y_dev, y_pred,average='weighted')
    precision = metrics.precision_score(y_dev, y_pred,average='weighted')
    recall = metrics.recall_score(y_dev, y_pred,average='weighted')
    
    print(f"f1 = {f1}")
    print(f"loss = {None}")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    
    
    with open(f"{language}/GaussianNB/test_results.txt", 'w') as f:
        f.write(f"f1 = {f1}\n")
        f.write(f"loss = {None}\n")
        f.write(f"precision = {precision}\n")
        f.write(f"recall = {recall}\n")
    
    print(f"[INFO] Saved {language}/GaussianNB/test_results.txt")
    f.close()


    
    print('=======   MultinomialNB   =========')

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predict Class
    y_pred = classifier.predict(X_dev)

    # Accuracy 
    accuracy = metrics.accuracy_score(y_dev, y_pred)
    f1 = metrics.f1_score(y_dev, y_pred, average='macro')


    print(f'acc: {accuracy}     |  f1_score: {f1}')
    print(metrics.classification_report(y_dev, y_pred, target_names=unique_label))
    
    if not os.path.exists(f"{language}/MultinomialNB"):
        os.makedirs(f"{language}/MultinomialNB")
        
    acc = metrics.accuracy_score(y_dev, y_pred)
    f1 = metrics.f1_score(y_dev, y_pred,average='weighted')
    precision = metrics.precision_score(y_dev, y_pred,average='weighted')
    recall = metrics.recall_score(y_dev, y_pred,average='weighted')
    
    print(f"f1 = {f1}")
    print(f"loss = {None}")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    
    
    with open(f"{language}/MultinomialNB/test_results.txt", 'w') as f:
        f.write(f"f1 = {f1}\n")
        f.write(f"loss = {None}\n")
        f.write(f"precision = {precision}\n")
        f.write(f"recall = {recall}\n")
    
    print(f"[INFO] Saved {language}/MultinomialNB/test_results.txt")
    f.close()
    

    print('=======   KNeighborsClassifier   =========')

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    # Predict Class
    y_pred = classifier.predict(X_dev)

    # Accuracy 
    accuracy = metrics.accuracy_score(y_dev, y_pred)
    f1 = metrics.f1_score(y_dev, y_pred, average='macro')


    print(f'acc: {accuracy}     |  f1_score: {f1}')
    print(metrics.classification_report(y_dev, y_pred, target_names=unique_label))
    
    if not os.path.exists(f"{language}/KNeighborsClassifier"):
        os.makedirs(f"{language}/KNeighborsClassifier")
        
    acc = metrics.accuracy_score(y_dev, y_pred)
    f1 = metrics.f1_score(y_dev, y_pred,average='weighted')
    precision = metrics.precision_score(y_dev, y_pred,average='weighted')
    recall = metrics.recall_score(y_dev, y_pred,average='weighted')
    
    print(f"f1 = {f1}")
    print(f"loss = {None}")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    
    
    with open(f"{language}/KNeighborsClassifier/test_results.txt", 'w') as f:
        f.write(f"f1 = {f1}\n")
        f.write(f"loss = {None}\n")
        f.write(f"precision = {precision}\n")
        f.write(f"recall = {recall}\n")
    
    print(f"[INFO] Saved {language}/KNeighborsClassifier/test_results.txt")
    f.close()


    print('=======   MLPClassifier   =========')

    classifier = MLPClassifier(random_state=1, max_iter=300)
    classifier.fit(X_train, y_train)

    # Predict Class
    y_pred = classifier.predict(X_dev)

    # Accuracy 
    accuracy = metrics.accuracy_score(y_dev, y_pred)
    f1 = metrics.f1_score(y_dev, y_pred, average='macro')


    print(f'acc: {accuracy}     |  f1_score: {f1}')
    print(metrics.classification_report(y_dev, y_pred, target_names=unique_label))
    
    if not os.path.exists(f"{language}/MLPClassifier"):
        os.makedirs(f"{language}/MLPClassifier")
        
    acc = metrics.accuracy_score(y_dev, y_pred)
    f1 = metrics.f1_score(y_dev, y_pred,average='weighted')
    precision = metrics.precision_score(y_dev, y_pred,average='weighted')
    recall = metrics.recall_score(y_dev, y_pred,average='weighted')
    
    print(f"f1 = {f1}")
    print(f"loss = {None}")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    
    
    with open(f"{language}/MLPClassifier/test_results.txt", 'w') as f:
        f.write(f"f1 = {f1}\n")
        f.write(f"loss = {None}\n")
        f.write(f"precision = {precision}\n")
        f.write(f"recall = {recall}\n")
    
    print(f"[INFO] Saved {language}/MLPClassifier/test_results.txt")
    f.close()

    print('=======   XGBClassifier   =========')

    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)

    # Predict Class
    y_pred = classifier.predict(X_dev)

    # Accuracy 
    accuracy = metrics.accuracy_score(y_dev, y_pred)
    f1 = metrics.f1_score(y_dev, y_pred, average='macro')


    print(f'acc: {accuracy}     |  f1_score: {f1}')
    print(metrics.classification_report(y_dev, y_pred, target_names=unique_label))
    
    if not os.path.exists(f"{language}/XGBClassifier"):
        os.makedirs(f"{language}/XGBClassifier")
        
    acc = metrics.accuracy_score(y_dev, y_pred)
    f1 = metrics.f1_score(y_dev, y_pred,average='weighted')
    precision = metrics.precision_score(y_dev, y_pred,average='weighted')
    recall = metrics.recall_score(y_dev, y_pred,average='weighted')
    
    print(f"f1 = {f1}")
    print(f"loss = {None}")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    
    
    with open(f"{language}/XGBClassifier/test_results.txt", 'w') as f:
        f.write(f"f1 = {f1}\n")
        f.write(f"loss = {None}\n")
        f.write(f"precision = {precision}\n")
        f.write(f"recall = {recall}\n")
    
    print(f"[INFO] Saved {language}/XGBClassifier/test_results.txt")
    f.close()
    
    

    print('=======   SVC   =========')
    classifier = SVC(gamma='auto')
    classifier.fit(X_train, y_train)
    # Predict Class
    y_pred = classifier.predict(X_dev)

    # Accuracy 
    accuracy = metrics.accuracy_score(y_dev, y_pred)
    f1 = metrics.f1_score(y_dev, y_pred, average='micro')


    print(f'acc: {accuracy}     |  f1_score: {f1}')
    print(metrics.classification_report(y_dev, y_pred, target_names=unique_label))
    
    if not os.path.exists(f"{language}/SVC"):
        os.makedirs(f"{language}/SVC")
        
    acc = metrics.accuracy_score(y_dev, y_pred)
    f1 = metrics.f1_score(y_dev, y_pred,average='weighted')
    precision = metrics.precision_score(y_dev, y_pred,average='weighted')
    recall = metrics.recall_score(y_dev, y_pred,average='weighted')
    
    print(f"f1 = {f1}")
    print(f"loss = {None}")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    
    
    with open(f"{language}/SVC/test_results.txt", 'w') as f:
        f.write(f"f1 = {f1}\n")
        f.write(f"loss = {None}\n")
        f.write(f"precision = {precision}\n")
        f.write(f"recall = {recall}\n")
    
    print(f"[INFO] Saved {language}/XGBClassifier/test_results.txt")
    f.close()

