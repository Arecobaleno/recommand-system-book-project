from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import pandas as pd
import os


class ChurnPredWithGBDT:
    def __init__(self):
        self.file = "../telecom-churn/telecom-churn-prediction-data.csv"
        self.data = self.feature_transform()
        self.train, self.test = self.split_data()

    def isNone(self, value):
        if value == " " or value is None:
            return "0.0"
        else:
            return value

    def feature_transform(self):
        if not os.path.exists("../telecom-churn/new_churn.csv"):
            print("Start feature transform ...")
            feature_dict = {
                "gender": {"Male":"1","Female":"0"},
                "Partner": {"Yes":"1","No":"0"},
                "Dependents": {"Yes":"1","No":"0"},
                "PhoneService": {"Yes":"1","No":"0"},
                "MultipleLines": {"Yes":"1","No":"0","No phone service":"2"},
                "InternetService": {"DSL":"1","Fiber optic":"2","No":"0"},
                "OnlineSecurity": {"Yes":"1","No":"0","No internet service":"2"},
                "OnlineBackup": {"Yes":"1","No":"0","No internet service":"2"},
                "DeviceProtection": {"Yes":"1","No":"0","No internet service":"2"},
                "TechSupport": {"Yes":"1","No":"0","No internet service":"2"},
                "StreamingTV": {"Yes":"1","No":"0","No internet service":"2"},
                "StreamingMovies": {"Yes":"1","No":"0","No internet service":"2"},
                "Contract": {"Month-to-month":"0","One year":"1","Two year":"2"},
                "PaperlessBilling":  {"Yes":"1","No":"0"},
                "PaymentMethod":{
                    "Electronic check":"0",
                    "Mailed check": "1",
                    "Bank transfer (automatic)": "2",
                    "Credit card (automatic)":"3",
                },
                "Churn": {"Yes":"1","No":"0"},
            }
            fw = open("../telecom-churn/new_churn.csv","w")
            fw.write(
                "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,"
                "InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
                "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn\n"
            )
            for line in open(self.file, "r").readlines():
                if line.startswith("customerID"):
                    continue
                customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines,\
                InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV,\
                StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn \
                = line.strip().split(",")
                _list = list()
                _list.append(customerID)
                _list.append(self.isNone(feature_dict["gender"][gender]))
                _list.append(self.isNone(SeniorCitizen))
                _list.append(self.isNone(feature_dict["Partner"][Partner]))
                _list.append(self.isNone(feature_dict["Dependents"][Dependents]))
                _list.append(self.isNone(tenure))
                _list.append(self.isNone(feature_dict["PhoneService"][PhoneService]))
                _list.append(self.isNone(feature_dict["MultipleLines"][MultipleLines]))
                _list.append(self.isNone(feature_dict["InternetService"][InternetService]))
                _list.append(self.isNone(feature_dict["OnlineSecurity"][OnlineSecurity]))
                _list.append(self.isNone(feature_dict["OnlineBackup"][OnlineBackup]))
                _list.append(self.isNone(feature_dict["DeviceProtection"][DeviceProtection]))
                _list.append(self.isNone(feature_dict["TechSupport"][TechSupport]))
                _list.append(self.isNone(feature_dict["StreamingTV"][StreamingTV]))
                _list.append(self.isNone(feature_dict["StreamingMovies"][StreamingMovies]))
                _list.append(self.isNone(feature_dict["Contract"][Contract]))
                _list.append(self.isNone(feature_dict["PaperlessBilling"][PaperlessBilling]))
                _list.append(self.isNone(feature_dict["PaymentMethod"][PaymentMethod]))
                _list.append(self.isNone(MonthlyCharges))
                _list.append(self.isNone(TotalCharges))
                _list.append(self.isNone(feature_dict["Churn"][Churn]))
                fw.write(",".join(_list))
                fw.write("\n")
            return pd.read_csv("../telecom-churn/new_churn.csv")
        else:
            return pd.read_csv("../telecom-churn/new_churn.csv")

    def split_data(self):
        train, test = train_test_split(self.data, test_size=0.1, random_state=40)
        return train, test

    def train_model(self):
        print("Start Train Model ... ")
        label = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.train.columns if x not in [label, ID]]
        x_train = self.train[x_columns]
        y_train = self.train[label]
        gbdt = GradientBoostingClassifier(learning_rate=0.1, n_estimators=200, max_depth=5)
        gbdt.fit(x_train, y_train)
        return gbdt

    def evaluate(self, gbdt):
        label = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.train.columns if x not in [label, ID]]
        x_test = self.test[x_columns]
        y_test = self.test[label]
        y_pre = gbdt.predict_proba(x_test)
        new_y_pre = list()
        for y in y_pre:
            new_y_pre.append(1 if y[1] > 0.5 else 0)
        mse = mean_squared_error(y_test, new_y_pre)
        print("MSE: %.4f" % mse)
        accuracy = metrics.accuracy_score(y_test.values, new_y_pre)
        print("Accuracy: %.4g" % accuracy)


if __name__ == "__main__":
    pred = ChurnPredWithGBDT()
    gbdt = pred.train_model()
    pred.evaluate(gbdt)