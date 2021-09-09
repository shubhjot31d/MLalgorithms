import csv
import random
import math
import matplotlib.pyplot as plt
import operator

def loaddataset(filename,split,trainingset=[],testset=[]):
    with open(filename,'r')as csvfile:
        lines=csv.reader(csvfile)
        dataset=list(lines)
        for x in range(len(dataset)):
            for y in range(8):
                dataset[x][y]=float(dataset[x][y])
            if random.random()<split:
                trainingset.append(dataset[x])
            else:
                testset.append(dataset[x])



def load_folds(folds,x,training_total,testing=[],training=[]):
       length=len(training_total)
       w=length//folds
       if x==0:
           for j in range(w):
               testing.append(training_total[j])
           for j in range(w+1,length):
               training.append(training_total[j])

       elif x==1:
           for j in range(w+1,2*w):
               testing.append(training_total[j])
           for j in range(w):
               training.append(training_total[j])
           for j in range(2*w,length):
               training.append(training_total[j])

       elif x==2:
           for j in range((2*w)+1,length):
               testing.append(training_total[j])
           for j in range(2*w):
               training.append(training_total[j])




def euclidiandistance(instance1,instance2,length):
    distance=0
    for x in range(length):
        distance+=pow((instance1[x]-instance2[x]),2)

    return math.sqrt(distance)




def get_neighbour(training,testinstance,K):
    distance=[]
    length=8
    for x in range(len(training)):
        dist = euclidiandistance(testinstance,training[x],length)
        distance.append((training[x],dist))
    distance.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in range(K):
        neighbors.append(distance[x][0])
    return neighbors


def response(neighbors):
    classvotes={}
    for x in range(len(neighbors)):
        response=neighbors[x][-1]
        if response in classvotes:
            classvotes[response]+=1
        else:
            classvotes[response]=1

    sortedvotes=sorted(classvotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedvotes[0][0]



def getaccuracy(testing,predictions):
    correct=0
    for x in range(len(testing)):
        if testing[x][-1] == predictions[x]:
            correct+=1
    return (correct/float(len(testing))*100.0)



trainingset_total=[]
testset=[]
split=0.70
folds=3
crossvalid=[]
trainingset=[]
loaddataset('liver.csv',split,trainingset_total,testset)
accuracies={}

for K in range(1,10):
    accuracy_ct = 0
    for k in range(folds):
        load_folds(folds,k,trainingset_total,crossvalid,trainingset)
        predictions=[]

        for y in range(len(crossvalid)):
            neighbors=get_neighbour(trainingset,crossvalid[y],K)
            result=response(neighbors)
            predictions.append(result)

        accuracy=getaccuracy(crossvalid,predictions)
        accuracy_ct+=accuracy
    accuracies[K]=(accuracy_ct/folds)

max_accuracy_K=max(accuracies.items(),key=operator.itemgetter(1))[0]
print ('Best value of K using cross validation is : ',max_accuracy_K)



loaddataset('liver.csv',split,trainingset_total,testset)
print('Train set : '+repr(len(trainingset_total)))
print('test set : '+repr(len(testset)))

#generate predictions
predictions=[]
k=max_accuracy_K
for x in range(len(testset)):
    neighbors=get_neighbour(trainingset_total,testset[x],k)
    result=response(neighbors)
    predictions.append(result)
    print('>predicted='+repr(result)+',actual='+repr(testset[x][-1]))
accuracy=getaccuracy(testset,predictions)
#plot_graph(testset,predictions)
print('Accuracy : '+repr(accuracy)+'%')



















