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

    def updateHostSince(self):
        dateDfOld = pd.to_datetime(self._dataFrame["host_since"], infer_datetime_format=True)
        currentDateTime = pd.to_datetime("now")
        diffDays = currentDateTime - dateDfOld
        diffDays = diffDays / np.timedelta64(1, 'D')
        self._dataFrame["host_since"] = diffDays

    def updateTrueFalseColumns(self,columnName):
        dataFrame = self._dataFrame[columnName].fillna('f') # fill the nan values with False values
        trueFalseDictionary = {'t':1 ,'f':0}
        # dataFrame.map({'t':1 ,'f':0})
        self._dataFrame[columnName] = dataFrame.apply(lambda x: trueFalseDictionary[x])

    def changePercentageToInt(self,columnName):
        dataFrame = self._dataFrame[columnName].fillna('00%')  # fill the nan values with 0 percent values values
        self._dataFrame[columnName] = dataFrame.apply(lambda x: int(float(x.split("%")[0])))
        pass

    def changeListToLength(self,columnName):
        dataFrame = self._dataFrame[columnName].fillna("")
        self._dataFrame[columnName] = dataFrame.apply(lambda x: len(x))

    def convertResponseTime(self):
        convertDict = {
            "a few days or more":0,
            "within an hour":1,
            "within a few hours":2,
            "within a day":3,
            " ":4
        }
        dataFrame = self._dataFrame['host_response_time'].fillna(" ")
        self._dataFrame["host_response_time"] = dataFrame.apply(lambda x:int(convertDict[x]))

if __name__ == '__main__':
    csvPath = "dataset/listings.csv"
    csvObj = CSVHelper(csvPath,["host_since","host_response_time","host_response_rate"
        ,"host_acceptance_rate","host_verifications","host_has_profile_pic","host_identity_verified","amenities",
        "review_scores_cleanliness","review_scores_checkin","review_scores_communication","review_scores_location",
                                "review_scores_value"],"host_is_superhost")
    csvObj.updateHostSince()
    csvObj.updateTrueFalseColumns("host_has_profile_pic")
    csvObj.updateTrueFalseColumns("host_identity_verified")
    csvObj.changePercentageToInt("host_response_rate")
    csvObj.changePercentageToInt("host_acceptance_rate")
    csvObj.changeListToLength("amenities")
    csvObj.changeListToLength("host_verifications")
    csvObj.convertResponseTime()
    dataDictionary = csvObj.getTrainTestData(0.1)
    pass