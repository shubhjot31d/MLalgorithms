import csv
import random
import operator
import math
import matplotlib.pyplot as plt
testsetname=[]
def loaddataset(filename,split,trainingset=[],testset=[]):
    with open(filename,'r')as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)):
            for y in range(4):
                dataset[x][y]=float(dataset[x][y])
            if random.random()<split:
                trainingset.append(dataset[x])
            else:
                testset.append(dataset[x])
                testsetname.append(dataset[x][-1])



def euclidiandistance(instance1,instance2,length):
    distance=0
    for x in range(length):
        distance+=pow((instance1[x]-instance2[x]),2)

    return math.sqrt(distance)


def getneighbor(trainingset,testinstance,k):
    distance=[]
    length=len(testinstance)-1
    for x in range(len(trainingset)):
        dist=euclidiandistance(testinstance,trainingset[x],length)
        distance.append((trainingset[x],dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(k):
        neighbors.append(distance[x][0])
    return neighbors



def getresponse(neighbors):
    classvotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classvotes:
            classvotes[response]+=1
        else:
            classvotes[response]=1

    sortedvotes=sorted(classvotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedvotes[0][0]



def getaccuracy(testset,predictions):
    correct=0
    for x in range(len(testset)):
        if testset[x][-1] == predictions[x]:
            correct+=1
    return (correct/float(len(testset)))*100.0

def precision(testset,predictions,key):
    TP=0
    FP=0
    for x in range(len(testset)):
        if testset[x][-1]==key and predictions[x]==key:
            TP+=1

    for y in range(len(testset)):
        if testset[y][-1]!=key and predictions[y]==key:
            FP+=1
    return (TP/(TP+FP),TP,FP)



def recall(testset,predictions,key):
    TP=0
    FN=0
    for x in range(len(testset)):
        if testset[x][-1]==key and predictions[x]==key:
            TP+=1

    for y in range(len(testset)):
        if testset[y][-1]==key and predictions[y]!=key:
            FN+=1
    return (TP/(TP+FN),TP,FN)


def false_positive_rate(testset,predeictions,key):
    FP=0
    TN=0
    for x in range(len(testset)):
        if testset[x][-1]!=key and predictions[x]==key:
            FP+=1


    for y in range(len(testset)):
        if testset[y][-1]!=key and predictions[y]!=key:
            TN+=1
    return (FP/(FP+TN),FP,TN)



def false_negative_rate(testset,predictions,key):
    FN=0
    TP=0
    for x in range(len(testset)):
        if testset[x][-1]==key and predictions[x]!=key:
            FN+=1

    for y in range(len(testset)):
        if testset[y][-1]==key and predictions[y]==key:
            TP+=1
    return (FN/(FN+TP),FN,TP)









def plot_graph(testset,predictions):
    x_axis1=[]
    x_axis2=[]
    for i in range(len(predictions)):
        x_axis1.append(i)
    for i in range(len(x_axis1)):
        x_axis2.append(i)

        plt.plot(x_axis1,predictions,label="Predicted")
        plt.plot(x_axis1,testsetname,label="Expexted")
        plt.legend()
        plt.show()





trainingset=[]
testset=[]
split=0.70
loaddataset('iris.csv',split,trainingset,testset)
print('Train set : '+repr(len(trainingset)))
print('test set : '+repr(len(testset)))

#generate predictions
predictions=[]
k=5
for x in range(len(testset)):
    neighbors=getneighbor(trainingset,testset[x],k)
    result=getresponse(neighbors)
    predictions.append(result)
    print('>predicted='+repr(result)+',actual='+repr(testset[x][-1]))
accuracy=getaccuracy(testset,predictions)
#plot_graph(testset,predictions)
print('Accuracy : '+repr(accuracy)+'%')


#PRECISION
preci_a,TP_a,FP_a= precision(testset,predictions,'setosa')
preci_b,TP_b,FP_b= precision(testset,predictions,'versicolor')
preci_c,TP_c,FP_c= precision(testset,predictions,'virginica')

TP=TP_a+TP_b+TP_c
FP=FP_a+FP_b+TP_c
print('Precision for g class = '+repr(preci_a))
print('Precision for b class = '+repr(preci_b))
print('Precision for c class = '+repr(preci_c))
print('Precision for the model = '+repr(TP/(TP+FP)))

#RECALL

recall_a,TP_a,FN_a= recall(testset,predictions,'setosa')
recall_b,TP_b,FN_b= recall(testset,predictions,'versicolor')
recall_c,TP_c,FN_c= recall(testset,predictions,'virginica')

TP=TP_a+TP_b+TP_c
FN=FN_a+FN_b+FN_c
print('Recall for setosa class = '+repr(recall_a))
print('Recall for versicolor class = '+repr(recall_b))
print('Recall for  virginica class = '+repr(recall_c))
print('Recall for the model = '+repr(TP/(TP+FN)))


#FALSE_POSITIVE_RATE
false_pos_a,FP_a,TN_a= false_positive_rate(testset,predictions,'setosa')
false_pos_b,FP_b,TN_b= false_positive_rate(testset,predictions,'versicolor')
false_pos_c,FP_c,TN_c= false_positive_rate(testset,predictions,'virginica')

FP=FP_a+FP_b+FP_c
TN=TN_a+TN_b+TN_c
print('false pos rate for setosa class = '+repr(false_pos_a))
print('false pos rate for versicolor class = '+repr(false_pos_b))
print('false pos rate for virginica class = '+repr(false_pos_c))
print('false pos rate for the model = '+repr(FP/(TN+FP)))

#FALSE_NEGATIVE_RATE
false_neg_a,FN_a,TP_a= false_negative_rate(testset,predictions,'setosa')
false_neg_b,FN_b,TP_b= false_negative_rate(testset,predictions,'versicolor')
false_neg_c,FN_c,TP_c= false_negative_rate(testset,predictions,'virginica')

FN=FN_a+FN_b+FN_c
TP=TP_a+TP_b+TP_c
print('false neg rate for setosa class = '+repr(false_neg_a))
print('false neg rate for versicolor class = '+repr(false_neg_b))
print('false neg rate for virginica class = '+repr(false_neg_c))
print('false neg rate for the model = '+repr(FN/(TP+FN)))










