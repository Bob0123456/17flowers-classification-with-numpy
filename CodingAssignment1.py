
# coding: utf-8

# In[1]:


import os
import numpy as np
from copy import deepcopy
from random import shuffle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[2]:


# Data Splitting
path = 'flowers17/feats/'

img_shuffle = []
img_set = []
class_counter = 0
counter = 0
for filename in sorted(os.listdir(path)):
    img_set.append(filename)
    counter += 1
    if counter == 80:
        shuffle(img_set)
        new_class = [deepcopy(img_set[0:40]), deepcopy(img_set[40:60]), deepcopy(img_set[60:80])]
        img_shuffle.append(new_class)
        img_set = []
        class_counter += 1
        counter = 0

flower_feats = deepcopy(img_shuffle)
for i in range(len(img_shuffle)):
    for j in range(len(img_shuffle[i])):
        for k in range(len(img_shuffle[i][j])):
            file = np.load(path + '/' + img_shuffle[i][j][k])
            flower_feats[i][j][k] = file


# In[3]:


def train_new_models(flower_feats, c):
    binary_svm = []
    for i in range(17):
        clf = SVC(C=c, kernel='linear', probability=True)
        train_x = []
        train_y = []
        for j in range(17):
            if i == j:
                train_x += flower_feats[j][0]
                train_y += [1]*len(flower_feats[j][0])
            else:
                train_x += flower_feats[j][0]
                train_y += [0]*len(flower_feats[j][0])
        clf.fit(train_x, train_y)
        binary_svm.append(clf)
    return binary_svm


# In[4]:


# Training model
binary_svm = train_new_models(flower_feats, 1.0)


# In[5]:


def predict_class(all_svm, features):
    class_prediction = []
    for i in range(len(all_svm)):
        current_prob = all_svm[i].predict_proba(features)
        if len(class_prediction) == 0:
            for probabilities in current_prob:
                class_prediction.append((i, probabilities[1]))
        else:
            for j in range(len(current_prob)):
                if class_prediction[j][1] < current_prob[j][1]:
                    class_prediction[j] = (i, current_prob[j][1])
                
    output = [prediction[0] for prediction in class_prediction]
    return output
        


# In[6]:


def validation_score(all_svm, flower_feats):
    y_predict = []
    y_true = []
    for i in range(len(flower_feats)):
        prediction = predict_class(all_svm, flower_feats[i][1])
        y_predict += prediction
        y_true += [i] * len(prediction)
    
    return accuracy_score(y_true, y_predict)


# In[7]:


# Testing validation using c = 1
validation_score(binary_svm, flower_feats)


# In[8]:


# C values to use
c_values = [0.01, 0.1, 0.1**0.5, 1, 10**0.5, 10, 100**0.5]


# In[9]:


# Finding best C
best_c = (0,0)
for c in c_values:
    test_svm = train_new_models(flower_feats, c)
    c_score = validation_score(test_svm, flower_feats)
    print('c value:{}    score:{}'.format(c, c_score))
    if c_score > best_c[1]:
        best_c = (c, c_score)
print(best_c)


# In[10]:


def train_final_models(flower_feats, c):
    binary_svm = []
    for i in range(17):
        clf = SVC(C=c, kernel='linear', probability=True)
        train_x = []
        train_y = []
        for j in range(17):
            if i == j:
                train_x += flower_feats[j][0]
                train_x += flower_feats[j][1]
                train_y += [1]*len(flower_feats[j][0])
                train_y += [1]*len(flower_feats[j][1])
            else:
                train_x += flower_feats[j][0]
                train_x += flower_feats[j][1]
                train_y += [0]*len(flower_feats[j][0])
                train_y += [0]*len(flower_feats[j][1])
        clf.fit(train_x, train_y)
        binary_svm.append(clf)
    return binary_svm


# In[11]:


def test_score(all_svm, flower_feats):
    y_predict = []
    y_true = []
    for i in range(len(flower_feats)):
        prediction = predict_class(all_svm, flower_feats[i][2])
        y_predict += prediction
        y_true += [i] * len(prediction)
        img_shown = 0
        for j in range(len(prediction)):
            if prediction[j] != i:
                print('{} wrongly classified for class {}'.format(img_shuffle[i][2][j],i))
    
    return accuracy_score(y_true, y_predict)


# In[12]:


# Training with training and validation set
final_svm = train_final_models(flower_feats, best_c[0])
# Scoring the final model
test_score(final_svm, flower_feats)


# In[14]:


# Saving train, val, test data
save_path = 'train_val_test'
counter = 0
for splits in flower_feats:
    np.save(save_path + '/' + 'train_class{}'.format(counter), splits[0])
    np.save(save_path + '/' + 'val_class{}'.format(counter), splits[1])
    np.save(save_path + '/' + 'test_class{}'.format(counter), splits[2])
    counter += 1

