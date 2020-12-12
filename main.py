from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from csvHelper import CSVHelper
from models import modelTrainHelper
import matplotlib.pyplot as plt

def multibarplot(data,title,filePath=None):
    """
    plot the error between the X vs Y and Z
    :param x: horizontal axis data
    :param y: y axis data usually mean error here
    :param z: z axis data usually Std error here
    :return:
    """
    plt.rc("font", size = 18)
    plt.title(title)
    plt.rcParams["figure.constrained_layout.use"] = True
    # for x,y,z in data:
    plt.errorbar(data[0], data[1], yerr=data[2], linewidth=3,label='Error Bar plot')
    plt.xlabel("Number Of Neighbours")
    plt.ylabel("F1 Score")
    plt.legend(bbox_to_anchor=(1.12, 1.1),loc='upper right')
    if filePath is None:
        plt.show()
    else:
        plt.savefig(filePath)

if __name__ == '__main__':
    datasetPath = "dataset/listings.csv"
    dataColsToUse = ["host_since","host_response_time","host_response_rate"
        ,"host_acceptance_rate","host_verifications","host_has_profile_pic","host_identity_verified","amenities",
        "review_scores_cleanliness","review_scores_checkin","review_scores_communication","review_scores_location",
        "review_scores_value","review_scores_accuracy","maximum_minimum_nights","minimum_maximum_nights",
        "minimum_nights_avg_ntm","maximum_nights_avg_ntm"]
    gtCol = "host_is_superhost"
    csvObj = CSVHelper(datasetPath,dataColsToUse,gtCol)
    csvObj.normaliseData()
    dataDict = csvObj.getTrainTestData(0.1)
    numNeighoursToUse = [3,5,7,10,15,100]
    trainingHelper = modelTrainHelper()
    meanTotal = []
    stdTotal = []
    for numNeighbour in numNeighoursToUse:
        model = KNeighborsClassifier(numNeighbour)
        xTrain = dataDict['Xtrain']
        yTrain = dataDict['yTrain']
        mean,std = trainingHelper.getCrossValScore(model,xTrain,yTrain)
        meanTotal.append(mean)
        stdTotal.append(std)
    multibarplot([numNeighoursToUse,meanTotal,stdTotal],"Error bar plot for KNN")