from sklearn.model_selection import KFold
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd

class KFoldValidation(object):

    def __init__(self, model, folds):
        self.model = model
        self.fold_count = folds
        self.folds = KFold(n_splits=folds)
        self.results = {v: 0 for v in ['accuracy','precision','recall','f1_score']}


    def validate(self,x,y):
        print('start')
        for r in self.results.keys():
            self.results[r] = 0
           
        
        folds = self.folds.split(x,y)
        for training_index, test_index in folds:
            x_train = x.iloc[training_index.tolist()]
            x_test = x.iloc[test_index.tolist()]
            y_train = y.iloc[training_index.tolist()]
            y_test = y.iloc[test_index.tolist()]
            prediction = self.runModel(x_train,y_train,x_test,y_test)
            self.results['accuracy'] += accuracy_score(y_test, prediction)
            self.results['precision'] += precision_score(y_test, prediction, average='weighted')
            self.results['recall'] += recall_score(y_test,prediction, average='weighted')
            self.results['f1_score'] += f1_score(y_test, prediction, average='weighted')

        self.printResults()
        print('finish')


    def runModel(self,x_train,y_train,x_test,y_test):
        self.model.fit(x_train,y_train)

        predictions = self.model.predict(x_test)
        return predictions

    def printResults(self):
        for k in self.results.keys():
            print("{0} is: {1}".format(k,self.results[k] / self.fold_count))