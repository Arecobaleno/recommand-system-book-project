import pandas as pd

"""
基于地域和热度的酒店推荐
"""
class RecBasedAH:
    def __init__(self, path=None, addr="朝阳区", type="score", k=10, sort=False):
        self.path = path
        self.addr = addr
        self.type = type
        self.k = k
        self.sort = sort
        self.data = self.load_mess()

    def load_mess(self):
        data = pd.read_csv(self.path, header=0, sep=",",encoding='GBK')
        return data[data["addr"] == self.addr]

    def recommend(self):
        if self.type in ["score","comment_num","lowest_price","decoration_time","open_time"]:
            data = self.data.sort_values(by=[self.type, "lowest_price"], ascending=self.sort)[:self.k]
            return dict(data.filter(items=["name", self.type]).values)
        elif self.type == "combine":
            data = self.data.filter(items=["name","score","comment_num","decoration_time","open_time","lowest_price"])
            data["decoration_time"] = data["decoration_time"].apply(lambda x:int(x)-2018)
            data["open_time"] = data["open_time"].apply(lambda x:2018-int(x))
            for col in data.keys():
                if col != "name":
                    data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
                data[self.type] = 1 * data["score"] + 2 * data["comment_num"] + 0.5 * data["decoration_time"] +\
                    0.5 * data["open_time"] + 1.5 * data["lowest_price"]
                data = data.sort_values(by=self.type, ascending=self.sort)[:self.k]
                return dict(data.filter(items=["name",self.type]).values)


if __name__ == "__main__":
    path = "../hotel-mess/hotel-mess.csv"
    rbah = RecBasedAH(path,type="combine",k=10,sort=False)
    print(rbah.recommend())