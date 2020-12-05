from sklearn.model_selection import cross_val_score
import numpy as np

class modelTrainHelper:
    """
    this class is a helper class which would help us train and test different sklearn models for our assignment
    """
    def getCrossValScore(self,model,X,y,cv=5):
        """
        General function to do the cross validation on our classification model
        :param model: Fresh model instance of our model
        :param X: whole training data.
        :param y: output of the data.
        :param cv: fraction to choose test data.
        :return:
        """
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
        meanF1 = np.array(scores).mean()
        stdF1  = np.array(scores).std()

        return meanF1,stdF1