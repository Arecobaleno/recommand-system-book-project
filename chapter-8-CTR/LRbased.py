from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import pandas as pd


class ChurnPredWithLR:
    def __init__(self):
        self.file = "../telecom-churn/new_churn.csv"
        self.data = self.load_data()
        self.train, self.test = self.split()

    def load_data(self):
        data = pd.read_csv(self.file)
        labels = list(data.keys())
        fDict = dict()
        for f in labels:
            if f not in ['customerID','tenure','MonthlyCharges','TotalCharges','Churn']:
                fDict[f] = sorted(list(data.get(f).unique()))
        fw = open("../telecom-churn/one_hot_churn.csv","w")
        fw.write("customerID,")
        for i in range(1,47):
            fw.write('f_%s,' % i)
        fw.write("Churn\n")
        for line in data.values:
            list_line = list(line)
            list_result = list()
            for i in range(0, list_line.__len__()):
                if labels[i] in ['customerID','tenure','MonthlyCharges','TotalCharges','Churn']:
                    list_result.append(list_line[i])
                else:
                    arr = [0] * fDict[labels[i]].__len__()
                    ind = fDict[labels[i]].index(list_line[i])
                    arr[ind] = 1
                    for one in arr:
                        list_result.append(one)
                    #list_result.append(arr)
            fw.write(",".join([str(f) for f in list_result]) + "\n")
        fw.close()
        return pd.read_csv("../telecom-churn/one_hot_churn.csv")

    def split(self):
        train, test = train_test_split(self.data, test_size=0.1, random_state=40)
        return train, test

    def train_model(self):
        print("Start Train Model ...")
        label = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.train.columns if x not in [ID, label]]
        x_train = self.train[x_columns]
        y_train = self.train[label]
        lr = LogisticRegression(penalty='l2', tol=1e-4, fit_intercept=True)
        lr.fit(x_train, y_train)
        return lr

    def evaluate(self,lr,type):
        label = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.train.columns if x not in [ID, label]]
        x_test = self.test[x_columns]
        y_test = self.test[label]
        if type == 1:
            y_pred = lr.predict(x_test)
            new_y_pre = y_pred
        elif type == 2:
            y_pred = lr.predict_proba(x_test)
            new_y_pre = list()
            for y in y_pred:
                new_y_pre.append(1 if y[1]>0.5 else 0)
        accuracy = metrics.accuracy_score(y_test.values, new_y_pre)
        print(accuracy)


if __name__ == "__main__":
    pred = ChurnPredWithLR()
    lr = pred.train_model()
    pred.evaluate(lr,type=1)