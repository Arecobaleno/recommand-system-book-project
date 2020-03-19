import pandas as pd
import math


class RecBasedTag:
    def __init__(self):
        # 用户听过艺术家次数的文件
        self.user_rate_file = "../lastfm-2k/user_artists.dat"
        # 用户打标信息
        self.user_tag_file = "../lastfm-2k/user_taggedartists.dat"

        # 获取所有艺术家ID
        self.artistsAll = list(pd.read_table("../lastfm-2k/artists.dat", delimiter="\t")["id"].values)
        # 用户对艺术家的评分
        self.userRateDict = self.getUserRate()
        # 艺术家和标签的相关度
        self.artistsTagsDict = self.getArtistsTags()
        # 用户对每个标签打标的次数统计和每个标签被所有用户打标的次数统计
        self.userTagDict, self.tagUserDict = self.getUserTagNum()
        # 用户最终对每个标签的喜好程度
        self.userTagPre = self.getUserTagPre()


    def getUserRate(self):
        userRateDict = dict()
        fr = open(self.user_rate_file, "r", encoding="utf-8")
        for line in fr.readlines():
            if not line.startswith("userID"):
                userID, artistID, weight = line.split("\t")
                userRateDict.setdefault(int(userID),{})
                userRateDict[int(userID)][int(artistID)] = float(weight) / 10000
        return userRateDict

    def getUserTagNum(self):
        userTagDict = dict()
        tagUserDict = dict()
        for line in open(self.user_tag_file, "r", encoding="utf-8"):
            if not line.startswith("userID"):
                userID, artistID, tagID = line.strip().split("\t")[:3]
                if int(tagID) in tagUserDict.keys():
                    tagUserDict[int(tagID)] += 1
                else:
                    tagUserDict[int(tagID)] = 1
                userTagDict.setdefault(int(userID),{})
                if int(tagID) in userTagDict[int(userID)].keys():
                    userTagDict[int(userID)][int(tagID)] += 1
                else:
                    userTagDict[int(userID)][int(tagID)] = 1
        return userTagDict, tagUserDict

    def getArtistsTags(self):  # 标签基因
        artistsTagsdict = dict()
        for line in open(self.user_tag_file, "r", encoding="utf-8"):
            if not line.startswith("userID"):
                artistID, tagID = line.strip().split("\t")[1:3]
                artistsTagsdict.setdefault(int(artistID),{})
                artistsTagsdict[int(artistID)][int(tagID)] = 1
        return artistsTagsdict

    def getUserTagPre(self):  # 用户对标签最终兴趣度
        userTagPre = dict()
        userTagCount = dict()
        Num = len(open(self.user_tag_file, "r", encoding="utf-8").readlines())
        for line in open(self.user_tag_file, "r", encoding="utf-8").readlines():
            if not line.startswith("userID"):
                userID, artistID, tagID = line.strip().split("\t")[:3]
                userTagPre.setdefault(int(userID),{})
                userTagCount.setdefault(int(userID),{})
                rate_ui = (self.userRateDict[int(userID)][int(artistID)] if int(artistID) in self.userRateDict[int(userID)].keys()
                           else 0)
                if int(tagID) not in userTagPre[int(userID)].keys():
                    userTagPre[int(userID)][int(tagID)] = rate_ui * self.artistsTagsDict[int(artistID)][int(tagID)]
                    userTagCount[int(userID)][int(tagID)] = 1
                else:
                    userTagPre[int(userID)][int(tagID)] += rate_ui * self.artistsTagsDict[int(artistID)][int(tagID)]
                    userTagCount[int(userID)][int(tagID)] += 1
        for userID in userTagPre.keys():
            for tagID in userTagPre[userID].keys():
                tf_ut = self.userTagDict[int(userID)][int(tagID)]/sum(self.userTagDict[int(userID)].values())
                idf_ut = math.log(Num * 1.0/(self.tagUserDict[int(tagID)]+1))
                userTagPre[userID][tagID] = userTagPre[userID][tagID]/userTagCount[userID][tagID] * tf_ut * idf_ut
        return userTagPre

    def recommendForUser(self, user, K, flag=True):
        userArtistPreDict = dict()
        for artist in self.artistsAll:
            if int(artist) in self.artistsTagsDict.keys():
                for tag in self.userTagPre[int(user)].keys():
                    rate_ut = self.userTagPre[int(user)][int(tag)]
                    rel_it = (0 if tag not in self.artistsTagsDict[int(artist)].keys()
                              else self.artistsTagsDict[int(artist)][tag])
                    if artist in userArtistPreDict.keys():
                        userArtistPreDict[int(artist)] += rate_ut * rel_it
                    else:
                        userArtistPreDict[int(artist)] = rate_ut * rel_it
        newUserArtistPreDict = dict()
        if flag:
            for artist in userArtistPreDict.keys():
                if artist not in self.userRateDict[int(user)].keys():
                    newUserArtistPreDict[artist] = userArtistPreDict[int(artist)]
            return sorted(newUserArtistPreDict.items(), key = lambda y:y[1], reverse=True)[:K]
        else:
            # 用来效果评估
            return sorted(userArtistPreDict.items(), key = lambda y:y[1], reverse=True)[:K]

    def evaluate(self, user):
        K = len(self.userRateDict[int(user)])
        recResult = self.recommendForUser(user, K=K, flag=False)
        count = 0
        for (artist, pre) in recResult:
            if artist in self.userRateDict[int(user)]:
                count += 1
        return count * 1.0 / K


if __name__ == "__main__":
    rbt = RecBasedTag()
    # print(rbt.recommendForUser("2",K=20))
    print(rbt.evaluate("2"))

