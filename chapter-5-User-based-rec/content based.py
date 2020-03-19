import pandas as pd
import json
import numpy as np
import math
import random


class DataProcessing:
    def __init__(self):
        pass

    def process(self):
        print('开始转换用户数据(users.dat)...')
        self.process_user_data()
        print('开始转换电影数据(movies.dat)...')
        self.process_movie_data()
        print('开始转换用户对电影评分数据(ratings.dat)')
        self.process_rating_data()
        print('Over!')

    def process_user_data(self, file='../ml-1m/users.dat'):
        fp = pd.read_table(file, sep='::', engine='python', names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
        fp.to_csv('../ml-1m/users.csv', index=False)

    def process_rating_data(self, file='../ml-1m/ratings.dat'):
        fp = pd.read_table(file, sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
        fp.to_csv('../ml-1m/ratings.csv', index=False)

    def process_movie_data(self, file='../ml-1m/movies.dat'):
        fp = pd.read_table(file, sep='::', engine='python', names=['MovieID', 'Title', 'Genres'])
        fp.to_csv('../ml-1m/movies.csv', index=False)

    def prepare_item_profile(self,file='../ml-1m/movies.csv'):  # 计算电影特征信息矩阵
        items = pd.read_csv(file)
        item_ids = set(items["MovieID"].values)
        self.item_dict = {}
        genres_all = list()
        # 将每个电影的类型放在item_dict中
        for item in item_ids:
            genres = items[items["MovieID"] == item]["Genres"].values[0].split("|")
            self.item_dict.setdefault(item, []).extend(genres)
            genres_all.extend(genres)
        self.genres_all = set(genres_all)
        # 将每个电影特征信息矩阵存放在self.item_matrix中
        self.item_matrix = {}
        for item in self.item_dict.keys():
            self.item_matrix[str(item)] = [0] * len(set(self.genres_all))
            for genre in self.item_dict[item]:
                index = list(set(genres_all)).index(genre)
                self.item_matrix[str(item)][index] = 1
        json.dump(self.item_matrix, open('../ml-1m/item_profile.json', 'w'))
        print("item信息计算完成，保存路径为'../ml-1m/item_profile.json'")

    def prepare_user_profile(self,file='../ml-1m/ratings.csv'):  # 计算用户偏好矩阵
        users = pd.read_csv(file)
        user_ids = set(users["UserID"].values)
        # 将user信息转换成dict
        users_rating_dict = {}
        for user in user_ids:
            users_rating_dict.setdefault(str(user),{})
        with open(file,"r") as fr:
            for line in fr.readlines():
                if not line.startswith("UserID"):
                    (user, item, rate) = line.split(",")[:3]
                    users_rating_dict[user][item] = int(rate)
        # 获取用户对每个类型下的哪些电影进行了评分
        self.user_matrix = {}
        for user in users_rating_dict.keys():
            score_list = users_rating_dict[user].values()
            avg = sum(score_list)/len(score_list)
            self.user_matrix[user] = []
            for genre in self.genres_all:
                score_all = 0.0
                score_len = 0
                for item in users_rating_dict[user].keys():
                    if genre in self.item_dict[int(item)]:
                        score_all += (users_rating_dict[user][item]-avg)
                        score_len += 1
                if score_len == 0:
                    self.user_matrix[user].append(0.0)
                else:
                    self.user_matrix[user].append(score_all/score_len)
        json.dump(self.user_matrix,open('../ml-1m/user_profile.json','w'))
        print("user信息计算完成，保存路径'../ml-1m/user_profile.json'")


class CBRecommend:
    def __init__(self,k):
        self.k = k  # 给用户推荐的item个数
        self.item_profile = json.load(open("../ml-1m/item_profile.json","r"))
        self.user_profile = json.load(open("../ml-1m/user_profile.json","r"))

    def get_none_score_item(self,user):
        items = pd.read_csv("../ml-1m/movies.csv")["MovieID"].values
        data = pd.read_csv("../ml-1m/ratings.csv")
        have_score_items = data[data["UserID"]==user]["MovieID"].values
        none_score_items = set(items)-set(have_score_items)
        return none_score_items

    def cosUI(self,user,item):
        Uia = sum(np.array(self.user_profile[str(user)]) * np.array(self.item_profile[str(item)]))
        Ua = math.sqrt(sum([math.pow(one,2) for one in self.user_profile[str(user)]]))
        Ia = math.sqrt(sum([math.pow(one,2) for one in self.item_profile[str(item)]]))
        return Uia / (Ua * Ia)

    def recommend(self,user):
        user_result = {}
        item_list = self.get_none_score_item(user)
        for item in item_list:
            user_result[item] = self.cosUI(user, item)
        if self.k is None:
            result = sorted(user_result.items(), key=lambda a:a[1], reverse=True)
        else:
            result = sorted(user_result.items(), key=lambda a:a[1], reverse=True)[:self.k]
        print(result)

    def evaluate(self):
        evas = []
        data = pd.read_csv("../ml-1m/ratings.csv")
        for user in random.sample([one for one in range(1,6041)],20):
            have_score_items = data



if __name__ == '__main__':
    # dp = DataProcessing()
    # dp.process()
    # dp.prepare_item_profile()
    # dp.prepare_user_profile()
    cb = CBRecommend(k=10)
    cb.recommend(1)
