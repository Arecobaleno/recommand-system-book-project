import pandas as pd
import pickle
import os
import random
import numpy as np
from math import exp
import time

class DataProcessing:
    def __init__(self):
        pass

    def get_pos_neg_item(self, file_path = "../ml-1m/ratings.csv"):
        if not os.path.exists("../ml-1m/lfm_items.dict"):
            self.items_dict_path = "../ml-1m/lfm_items.dict"
            self.uiscores = pd.read_csv(file_path)
            self.user_ids = set(self.uiscores["UserID"].values)
            self.item_ids = set(self.uiscores["MovieID"].values)
            self.items_dict = {user_id: self.get_one(user_id) for user_id in list(self.user_ids)}
            fw = open(self.items_dict_path, "wb")
            pickle.dump(self.items_dict, fw)
            fw.close()

    def get_one(self, user_id):
        print('为用户%s准备正向和负向数据...' % user_id)
        pos_item_ids = set(self.uiscores[self.uiscores['UserID'] == user_id]['MovieID'])
        neg_item_ids = self.item_ids ^ pos_item_ids  # “^”为异或符号
        neg_item_ids = list(neg_item_ids)[:len([pos_item_ids])]
        item_dict = {}
        for item in pos_item_ids:
            item_dict[item] = 1
        for item in neg_item_ids:
            item_dict[item] = 0
        return item_dict


class LFM:
    def __init__(self):
        self.class_count = 5
        self.iter_count = 5
        self.lr = 0.02
        self.lam = 0.01
        self._init_model()

    def _init_model(self):
        file_path = '../ml-1m/ratings.csv'
        pos_neg_path = '../ml-1m/lfm_items.dict'
        self.uiscores = pd.read_csv(file_path)
        self.user_ids = set(self.uiscores['UserID'].values)
        self.item_ids = set(self.uiscores['MovieID'].values)
        self.items_dict = pickle.load(open(pos_neg_path,'rb'))

        array_p = np.random.randn(len(self.user_ids),self.class_count)
        array_q = np.random.randn(len(self.item_ids),self.class_count)
        self.p = pd.DataFrame(array_p, columns=range(0, self.class_count), index=list(self.user_ids))
        self.q = pd.DataFrame(array_q, columns=range(0, self.class_count), index=list(self.item_ids))

    def _predict(self, user_id, item_id):
        p = np.mat(self.p.ix[user_id].values)
        q = np.mat(self.q.ix[item_id].values).T
        r = (p * q).sum()
        logit = 1.0 / (1 + exp(-r))
        return logit

    def _loss(self, user_id, item_id, y, step):
        e = y - self._predict(user_id, item_id)
        return e

    def _optimize(self, user_id, item_id, e):
        gradient_p = -e * self.q.ix[item_id].values
        l2_p = self.lam * self.p.ix[user_id].values
        delta_p = self.lr * (gradient_p + l2_p)

        gradient_q = -e * self.p.ix[user_id].values
        l2_q = self.lam * self.q.ix[item_id].values
        delta_q = self.lr * (gradient_q + l2_q)

        self.p.loc[user_id] -= delta_p
        self.q.loc[item_id] -= delta_q

    def train(self):
        for step in range(0,self.iter_count):
            time.sleep(30)
            for user_id, item_dict in self.items_dict.items():
                print('step: {}, user_id: {}'.format(step, user_id))
                item_ids = list(item_dict.keys())
                random.shuffle(item_ids)
                for item_id in item_ids:
                    e = self._loss(user_id, item_id, item_dict[item_id], step)
                    self._optimize(user_id, item_id, e)
            self.lr *= 0.9
        self.save()

    def save(self):
        f = open('../ml-1m/lfm.model','wb')
        pickle.dump((self.p, self.q), f)
        f.close()

    def load(self):
        f = open('../ml-1m/lfm.model','rb')
        self.p, self.q = pickle.load(f)
        f.close()

    def predict(self, user_id, top_n = 10):
        self.load()
        user_item_ids = set(self.uiscores[self.uiscores['UserID'] == user_id]['MovieID'])
        other_item_ids = self.item_ids ^ user_item_ids
        interest_list = [self._predict(user_id, item_id) for item_id in other_item_ids]
        candidates = sorted(zip(list(other_item_ids), interest_list), key= lambda x:x[1], reverse=True)
        return candidates[:top_n]

    def evaluate(self):
        self.load()
        users = random.sample(self.user_ids, 10)
        user_dict = {}
        for user in users:
            user_item_ids = set(self.uiscores[self.uiscores['UserID'] == user]['MovieID'])
            _sum = 0.0
            for item_id in user_item_ids:
                p = np.mat(self.p.ix[user].values)
                q = np.mat(self.q.ix[item_id].values).T
                _r = (p * q).sum()
                r = self.uiscores[(self.uiscores['UserID'] == user)&(self.uiscores['MovieID']==item_id)]["Rating"].values[0]
                _sum += abs(r - _r)
            user_dict[user] =  _sum/len(user_item_ids)
            print("user: {} AI: {}".format(user,user_dict[user]))
        return sum(user_dict.values()) / len(user_dict.keys())


if __name__ == '__main__':
    # dp = DataProcessing()
    # dp.get_pos_neg_item()
    lfm = LFM()
    #lfm.train()
    #print(lfm.predict(6027,10))
    print(lfm.evaluate())