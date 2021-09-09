#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[31]:


data=pd.read_csv("bank.csv")


# In[32]:


data


# In[4]:


import csv
import math
import random


# In[5]:


class DecisionTree():
    tree = {}

    def learn(self, training_set, attributes, target):
        self.tree = build_tree(training_set, attributes, target)


# In[6]:


class Node():
    value = ""
    children = []

    def __init__(self, val, dictionary):
        self.value = val
        if (isinstance(dictionary, dict)):
            self.children = dictionary.keys()


# In[7]:


def majorClass(attributes, data, target):

    freq = {}
    index = attributes.index(target)

    for tuple in data:
        if (tuple[index] in freq):
            freq[tuple[index]] += 1 
        else:
            freq[tuple[index]] = 1

    max = 0
    major = ""

    for key in freq.keys():
        if freq[key]>max:
            max = freq[key]
            major = key

    return major


# In[8]:


def entropy(attributes, data, targetAttr):

    freq = {}
    dataEntropy = 0.0

    i = 0
    for entry in attributes:
        if (targetAttr == entry):
            break
        i = i + 1

    i = i - 1

    for entry in data:
        if (entry[i] in freq):
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]]  = 1.0

    for freq in freq.values():
        dataEntropy += (-freq/len(data)) * math.log(freq/len(data), 2) 
        
    return dataEntropy


# In[9]:


def info_gain(attributes, data, attr, targetAttr):

    freq = {}
    subsetEntropy = 0.0
    i = attributes.index(attr)

    for entry in data:
        if (entry[i] in freq):
            freq[entry[i]] += 1.0
        else:
            freq[entry[i]]  = 1.0

    for val in freq.keys():
        valProb        = freq[val] / sum(freq.values())
        dataSubset     = [entry for entry in data if entry[i] == val]
        subsetEntropy += valProb * entropy(attributes, dataSubset, targetAttr)

    return (entropy(attributes, data, targetAttr) - subsetEntropy)


# In[10]:


def attr_choose(data, attributes, target):

    best = attributes[0]
    maxGain = 0;

    for attr in attributes:
        newGain = info_gain(attributes, data, attr, target) 
        if newGain>maxGain:
            maxGain = newGain
            best = attr

    return best


# In[11]:


def get_values(data, attributes, attr):

    index = attributes.index(attr)
    values = []

    for entry in data:
        if entry[index] not in values:
            values.append(entry[index])

    return values


# In[12]:


def get_data(data, attributes, best, val):

    new_data = [[]]
    index = attributes.index(best)

    for entry in data:
        if (entry[index] == val):
            newEntry = []
            for i in range(0,len(entry)):
                if(i != index):
                    newEntry.append(entry[i])
            new_data.append(newEntry)

    new_data.remove([])    
    return new_data


# In[13]:


def build_tree(data, attributes, target):
    
    vals = [record[attributes.index(target)] for record in data]
    default = majorClass(attributes, data, target)

    if not data or (len(attributes) - 1) <= 0:
        return default
    elif vals.count(vals[0]) == len(vals):
        return vals[0]
    else:
        best = attr_choose(data, attributes, target)
        tree = {best:{}}
    
        for val in get_values(data, attributes, best):
            new_data = get_data(data, attributes, best, val)
            newAttr = attributes[:]
            newAttr.remove(best)
            subtree = build_tree(new_data, newAttr, target)
            tree[best][val] = subtree
    
    return tree


# In[14]:


random.seed(42)
def run_decision_tree(data,attributes,K=10):

    target = attributes[-1]

    acc = []
    for k in range(K):
        random.shuffle(data)
        training_set = [x for i, x in enumerate(data) if i % K != k]
        test_set = [x for i, x in enumerate(data) if i % K == k]
        tree = DecisionTree()
        tree.learn( training_set, attributes, target )
        results = []

        for entry in test_set:
            tempDict = tree.tree.copy()
            result = ""
            while(isinstance(tempDict, dict)):
                root = Node(list(tempDict.keys())[0], tempDict[list(tempDict.keys())[0]])
                tempDict = tempDict[list(tempDict.keys())[0]]
                index = attributes.index(root.value)
                value = entry[index]
                if(value in tempDict.keys()):
                    child = Node(value, tempDict[value])
                    result = tempDict[value]
                    tempDict = tempDict[value]
                else:
                    result = "Null"
                    break
            if result != "Null":
                results.append(result == entry[-1])

        accuracy = float(results.count(True))/float(len(results))
        acc.append(accuracy)

    avg_acc = sum(acc)/len(acc)
    return results,avg_acc,acc


# In[15]:


subset = data[data.columns]
tuples = [tuple(x) for x in subset.to_numpy()]
attributes=list(data.columns)


# In[16]:


attributes


# In[17]:


accuracy=[]
for k in range(3,11):
    results,avg_acc,acc=run_decision_tree(tuples,attributes,k)
    accuracy.append(avg_acc)
    print(k,avg_acc)
print(accuracy)


# In[33]:


data=data.drop(['month','day'],axis=1)


# In[34]:


data


# In[35]:


subset = data[data.columns]
tuples = [tuple(x) for x in subset.to_numpy()]
attributes=list(data.columns)


# In[36]:


attributes


# In[37]:


results,avg_acc,acc=run_decision_tree(tuples,attributes)
print(avg_acc)
print(acc)


# In[38]:


data=data.drop('marital',axis=1)


# In[39]:


data


# In[40]:


subset = data[data.columns]
tuples = [tuple(x) for x in subset.to_numpy()]
attributes=list(data.columns)


# In[41]:


results,avg_acc,acc=run_decision_tree(tuples,attributes)
print(avg_acc)


# In[42]:


print(acc)


# In[ ]:




