from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from numpy import array

conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
sc = SparkContext(conf= conf)

# functions for convert csv file into numerical feature
def binary(YN):
    if (YN == 'Y'):
        return 1
    else:
        return 0

def mapEducation(degree):
    if (degree == 'BS'):
        return 1
    elif (degree == 'MS'):
        return 2
    elif (degree == 'PhD'):
        return 3
    else:
        return 0
    
# convert list of raw fields into labeled point for MLlib use
def CreateLabeledPoints(fields):
    yearsExperience = int(fields[0])
    employed = binary(fields[1])
    previousEmployers = int(fields[2])
    educationLevel = mapEducation(fields[3])
    topTier = binary(fields[4])
    interned = binary(fields[5])
    hiered = binary(fields[6])

    return LabeledPoint(hiered, array([yearsExperience, employed, previousEmployers, educationLevel, topTier, interned]))

# load data
rawdata = sc.textFile(r"E:\AI Projects\Spark\sampledata.csv")
header = rawdata.first()
rawdata = rawdata.filter(lambda x: x != header)

# split each line into a list based on comma delimiters
csvdata = rawdata.map(lambda x: x.split(","))

# convert lists to labeledPoints
trainingdata = csvdata.map(CreateLabeledPoints)

testCandidate = [array([10,1,3,1,0,0])]
testData = sc.parallelize(testCandidate)

# train model
model = DecisionTree.trainClassifier(trainingdata,
                                     numClasses= 2,
                                     categoricalFeaturesInfo={1:2, 3:4, 4:2, 5:2},
                                     impurity= 'gini',
                                     maxDepth= 5,
                                     maxBins= 32)

predictions = model.predict(testData)
print('Hire Predictions:')
results = predictions.collect()

for result in results:
    print(result)

print('learned classification tree model:')
print(model.toDebugString())