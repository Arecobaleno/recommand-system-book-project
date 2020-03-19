from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class GBDTWithLR:
    def __init__(self):
        self.file = "../telecom-churn/new_churn.csv"
        self.data = self.load_data()
        self.train, self.test = self.split()

    def load_data(self):
        return pd.read_csv(self.file)

    def split(self):
        train, test = train_test_split(self.data, test_size=0.1, random_state=40)
        return train, test

    def train_model(self):
        print("training")
        label = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.train.columns if x not in [ID, label]]
        x_train = self.train[x_columns]
        y_train = self.train[label]

        gbdt = GradientBoostingClassifier()
        gbdt.fit(x_train,y_train)

        gbdt_lr = LogisticRegression(max_iter=3000)
        enc = OneHotEncoder()
        enc.fit(gbdt.apply(x_train).reshape(-1,100))

        gbdt_lr.fit(enc.transform(gbdt.apply(x_train).reshape(-1,100)),y_train)
        return enc, gbdt, gbdt_lr

    def evaluate(self,enc,gbdt,gbdt_lr):
        print("evaluating")
        label = "Churn"
        ID = "customerID"
        x_columns = [x for x in self.train.columns if x not in [ID, label]]
        x_test = self.test[x_columns]
        y_test = self.test[label]

        gbdt_pred = gbdt.predict(x_test)
        print("GBDT accuracy: %.4g" % metrics.accuracy_score(y_test.values, gbdt_pred))

        gbdt_lr_pred = gbdt_lr.predict(enc.transform(gbdt.apply(x_test).reshape(-1,100)))
        print("GBDT_LR accuracy: %.4g" % metrics.accuracy_score(y_test.values, gbdt_lr_pred))


if __name__ == "__main__":
    new_model= GBDTWithLR()
    enc, gbdt, gbdt_lr = new_model.train_model()
    new_model.evaluate(enc, gbdt, gbdt_lr)