import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class CSVHelper:
    def __init__(self, pathToCsv, columnNames=None, gtColmunName=None):
        """
        load the csv
        :param pathToCsv: csv Path to load
        :param columnNames: sub column names to use
        :param gtColmunName: Groundtruth column name
        """
        self._dataFrame = pd.read_csv(pathToCsv)
        self.updateGtColumn(gtColmunName)
        self.updatetheDataframes(columnNames)
        self.dataDirectory = None

    def updatetheDataframes(self,columnsName):
        """
        name of columns to select from the dataframes
        :param columnsName: list of columns
        :return:
        """
        self._colNamesToUse  = columnsName

    def updateGtColumn(self,columnName):
        """
        name of the column to be used as GT
        :param columnName: name of the column to be used as GT
        :return:
        """
        self._gtColName = columnName

    def doTrainTestSplit(self,testSize=0.1):
        """
        get the train test data from the csv
        :param testSize: testSize
        :return:
        """
        if self._colNamesToUse is not None:
            X = np.asarray(self._dataFrame[self._colNamesToUse])
        else:
            X = np.asarray(self._dataFrame)

        assert self._gtColName is not None,"Please assign a gt column name"
        y = np.asarray(self._dataFrame[self._gtColName])

        Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size=testSize, random_state=42)
        self.dataDirectory = {"Xtrain":Xtrain,"Xtest":Xtest,"yTrain":yTrain,"yTest":yTest}

    def getTrainTestData(self, testSize=0.1,useOldData = True):
        if self.dataDirectory is None:
            self.doTrainTestSplit(testSize)
        if useOldData:
            return self.dataDirectory
        else:
            self.doTrainTestSplit(testSize)
            return self.dataDirectory

if __name__ == '__main__':
    csvPath = "dataset/listings.csv"
    csvObj = CSVHelper(csvPath,["host_since","host_response_time","host_response_rate"
        ,"host_acceptance_rate","host_verifications","host_has_profile_pic","host_identity_verified","amenities",
        "review_scores_cleanliness","review_scores_checkin","review_scores_communication","review_scores_location",
                                "review_scores_value","license"],"host_is_superhost")
    dataDictionary = csvObj.getTrainTestData(0.1)
    pass