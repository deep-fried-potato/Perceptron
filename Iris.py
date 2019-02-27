import random
class DataPoint:
    def __init__(self,attributes):
        self.attributes= [1] + [float(x) for x in attributes[:-1]]
        self.value = 1 if attributes[-1]=="Iris-versicolor" else -1
    def __str__(self):
        return str(self.attributes) + " : " +str(self.value)

def signum(a):
    return (a>0)-(a<0)

def dotprod(K, L):
    if len(K) != len(L):
        return 0
    return sum(i[0] * i[1] for i in zip(K, L))

def addlists(list1,list2):
    return [x1+x2 for (x1,x2) in zip(list1,list2)]

def subtract(List1,List2):
    return [x1 - x2 for (x1, x2) in zip(List1, List2)]

def scalarprod(scalar,vector):
    return [scalar*element for element in vector]

def MisclassifyCount(DataSet,weights):
    result=0
    for point in DataSet:
        result += abs(signum(dotprod(point.attributes,weights))-point.value)
    return str(result)+"/"+str(len(DataSet))

def gradient(DataSet,weights):
    result = [0]*len(DataSet[0].attributes)
    for point in DataSet:
        result=addlists(result,scalarprod((signum(dotprod(point.attributes,weights)) - point.value),point.attributes))
    return scalarprod(1/len(DataSet),result)

def Train(TrainingSet,eta):
    weights = [0]*len(DataSet[0].attributes)
    epoch=1
    while any(abs(grad)>0.01 for grad in gradient(TrainingSet,weights)):
        weights = subtract(weights,scalarprod(eta,gradient(TrainingSet,weights)))
        if epoch%100==0:
            print("Iteration: ",epoch," Misclassified: ",MisclassifyCount(TrainingSet,weights)," Gradient: ",gradient(TrainingSet,weights)," Weights: ",weights)
        epoch+=1
    print("Iteration: ",epoch," Misclassified: ",MisclassifyCount(TrainingSet,weights)," Gradient: ",gradient(TrainingSet,weights)," Weights: ",weights)
    return weights


fileobj = open("iris.data","r")
filedata = fileobj.readlines()
DataSet = [DataPoint([x for x in line.replace("\n","").split(",")]) for line in filedata if line.replace("\n","").split(",")[-1] not in ["","Iris-setosa"] ]
random.shuffle(DataSet)

TrainSize=70
eta=0.01
TrainingSet = DataSet[:TrainSize]
TestSet= DataSet[TrainSize:]

trained_weights=Train(TrainingSet,eta)
print("Test Error: ", MisclassifyCount(TestSet,trained_weights))
