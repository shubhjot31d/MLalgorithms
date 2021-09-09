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
        for x in range(len(dataset)-1):
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
split=0.677777778
loaddataset('iris1.csv',split,trainingset,testset)
print('Train set : '+repr(len(trainingset)))
print('test set : '+repr(len(testset)))

#generate predictions
predictions=[]
k=3
for x in range(len(testset)):
    neighbors=getneighbor(trainingset,testset[x],k)
    result=getresponse(neighbors)
    predictions.append(result)
    print('>predicted='+repr(result)+',actual='+repr(testset[x][-1]))
accuracy=getaccuracy(testset,predictions)
plot_graph(testset,predictions)
print('Accuracy : '+repr(accuracy)+'%')
