import os
import json
import random
import math

"""
第二章内容，搭建第一个推荐系统
"""


class FirstRec:
    """
    k 近邻用户个数
    n_items 为每个用户推荐的电影数
    """
    def __init__(self,file_path,seed,k,n_items):
        self.file_path = file_path
        self.users_1000 = self.__select_1000_users()
        self.seed = seed
        self.k = k
        self.n_items = n_items
        self.train,self.test = self._load_and_split_data()

    def __select_1000_users(self):
        print("随机选取1000个用户！")
        if os.path.exists("../data/train.json") and os.path.exists("../data/test.json"):
            return list()
        else:
            users = set()
            for file in os.listdir(self.file_path):
                one_path = "{}/{}".format(self.file_path,file)
                print("{}".format(one_path))
                with open(one_path, "r") as fp:
                    for line in fp.readlines():
                        if line.strip().endswith(":"):
                            continue
                        userID, _, _ = line.split(",")
                        users.add(userID)
            users_1000 = random.sample(list(users),1000)
            print(users_1000)
            return users_1000

    def _load_and_split_data(self):
        train = dict()
        test = dict()
        if os.path.exists("../data/train.json") and os.path.exists("../data/test.json"):
            print("从文件中加载训练集和测试集")
            train = json.load(open("../data/train.json"))
            test = json.load(open("../data/test.json"))
            print("从文件中加载数据完成")
        else:
            random.seed(self.seed)
            for file in os.listdir(self.file_path):
                one_path = "{}/{}".format(self.file_path,file)
                print("{}".format(one_path))
                with open(one_path,"r") as fp:
                    movieID = fp.readline().split(":")[0]
                    for line in fp.readlines():
                        if line.strip().endswith(":"):
                            movieID = line.split(":")[0]
                            continue
                        userID, rate, _ = line.split(",")
                        if userID in self.users_1000:
                            if random.randint(1,50) == 1:
                                test.setdefault(userID, {})[movieID] = int(rate)
                            else:
                                train.setdefault(userID, {})[movieID] = int(rate)
            print("加载数据到 ../data/train.json 和 ../data/test.json")
            json.dump(train, open("../data/train.json", "w"))
            json.dump(test, open("../data/test.json", "w"))
            print("加载完成")
        return train, test

    def pearson(self,rating1,rating2):
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        num = 0
        for key in rating1.keys():
            if key in rating2.keys():
                num += 1
                x = rating1[key]
                y = rating2[key]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += math.pow(x,2)
                sum_y2 += math.pow(y,2)
        if num == 0:
            return 0
        de = math.sqrt(sum_x2 - math.pow(sum_x,2)/num) * math.sqrt(sum_y2 - math.pow(sum_y,2)/num)
        if de == 0:
            return 0
        else:
            return (sum_xy - (sum_x * sum_y) / num) / de

    def recommend(self,userID):
        neighborUser = dict()
        for user in self.train.keys():
            if user != userID:
                distance = self.pearson(self.train[user],self.train[userID])
                neighborUser[user] = distance
        newNU = sorted(neighborUser.items(), key=lambda m: m[1], reverse= True)
        movies = dict()
        for (sim_user,sim) in newNU[:self.k]:
            for movieID in self.train[sim_user].keys():
                movies.setdefault(movieID,0)
                movies[movieID] += sim * self.train[sim_user][movieID]
        newMovies = sorted(movies.items(),key = lambda m:m[1],reverse=True)
        return newMovies

    def evaluate(self,num=30):
        print("开始计算准确率")
        precisions = list()
        random.seed(10)
        for userID in random.sample(self.test.keys(),num):
            hit = 0
            result = self.recommend(userID)[:self.n_items]
            for (item,rate) in result:
                if item in self.test[userID]:
                    hit += 1
            precisions.append(hit/self.n_items)
        return sum(precisions) / precisions.__len__()


if __name__ == "__main__":
    file_path = "../data/training_set"
    seed = 30
    k = 15
    n_items = 20
    f_rec = FirstRec(file_path,seed,k,n_items)
    print("算法的推荐准确率{}".format(f_rec.evaluate()))