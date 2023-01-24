import os
import pandas as pd
import tqdm

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

folder_name = 'SubtaskA/'


language_list = set([i.split('_')[0]  for i in os.listdir(folder_name) if '_new.tsv' in i ])
print(language_list)
for language in language_list:
    print('-------------------------------------------------')
    print(f'--------------Working on {language}-----------------')
    
    train_data = pd.read_csv(f'{folder_name}/{language}_train_new.tsv',sep='\t')
    dev_data = pd.read_csv(f'{folder_name}/{language}_dev_new.tsv',sep='\t')

    print(f' Training set size : {train_data.size}   Dev set size: {dev_data.size}')
    
    all_text_list  = train_data['tweet'].values.tolist()+dev_data['tweet'].values.tolist() 
    
    train_text,train_label = train_data['tweet'].values.tolist(),train_data['label'].values.tolist()
    dev_text,dev_label = dev_data['tweet'].values.tolist(),dev_data['label'].values.tolist()

    unique_label = train_data['label'].unique().tolist()
    
    # CountVectorizer
    count_vectorizer = CountVectorizer(analyzer='char_wb',ngram_range=(1, 3))
    count_vectorizer.fit_transform(all_text_list)
    
    X_train = count_vectorizer.transform(train_text).toarray()
    X_dev= count_vectorizer.transform(dev_text).toarray()
    
    y_train = []
    for i in train_label:
        y_train.append(unique_label.index(i))

    y_dev = []
    for i in dev_label:
        y_dev.append(unique_label.index(i))
        
    print(f'Sizes : {X_train.shape,X_dev.shape,len(y_train),len(y_dev)}')



    print('=======   GaussianNB   =========')

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predict Class
    y_pred = classifier.predict(X_dev)

    # Accuracy 
    accuracy = accuracy_score(y_dev, y_pred)
    f1 = f1_score(y_dev, y_pred, average='macro')


    print(f'acc: {accuracy}     |  f1_score: {f1}')
    print(classification_report(y_dev, y_pred, target_names=unique_label))



    print('=======   MultinomialNB   =========')

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predict Class
    y_pred = classifier.predict(X_dev)

    # Accuracy 
    accuracy = accuracy_score(y_dev, y_pred)
    f1 = f1_score(y_dev, y_pred, average='macro')


    print(f'acc: {accuracy}     |  f1_score: {f1}')
    print(classification_report(y_dev, y_pred, target_names=unique_label))


    print('=======   KNeighborsClassifier   =========')

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)

    # Predict Class
    y_pred = classifier.predict(X_dev)

    # Accuracy 
    accuracy = accuracy_score(y_dev, y_pred)
    f1 = f1_score(y_dev, y_pred, average='macro')


    print(f'acc: {accuracy}     |  f1_score: {f1}')
    print(classification_report(y_dev, y_pred, target_names=unique_label))


    print('=======   MLPClassifier   =========')

    classifier = MLPClassifier(random_state=1, max_iter=300)
    classifier.fit(X_train, y_train)

    # Predict Class
    y_pred = classifier.predict(X_dev)

    # Accuracy 
    accuracy = accuracy_score(y_dev, y_pred)
    f1 = f1_score(y_dev, y_pred, average='macro')


    print(f'acc: {accuracy}     |  f1_score: {f1}')
    print(classification_report(y_dev, y_pred, target_names=unique_label))

    print('=======   XGBClassifier   =========')

    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)

    # Predict Class
    y_pred = classifier.predict(X_dev)

    # Accuracy 
    accuracy = accuracy_score(y_dev, y_pred)
    f1 = f1_score(y_dev, y_pred, average='macro')


    print(f'acc: {accuracy}     |  f1_score: {f1}')
    print(classification_report(y_dev, y_pred, target_names=unique_label))

    print('=======   SVC   =========')
    classifier = SVC(gamma='auto')
    classifier.fit(X_train, y_train)
    # Predict Class
    y_pred = classifier.predict(X_dev)

    # Accuracy 
    accuracy = accuracy_score(y_dev, y_pred)
    f1 = f1_score(y_dev, y_pred, average='micro')


    print(f'acc: {accuracy}     |  f1_score: {f1}')
    print(classification_report(y_dev, y_pred, target_names=unique_label))


