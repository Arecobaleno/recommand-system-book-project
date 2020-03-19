import matplotlib.pyplot as plt
import numpy as np


class UserShow:
    def __init__(self):
        self.file_user = "../bookcrossings/BX-Users.csv"
        self.file_book = "../bookcrossings/BX-Books.csv"
        self.file_rate = "../bookcrossings/BX-Book-Ratings.csv"
        self.user_mess = self.loadUserData()
        self.book_mess = self.loadBookMess()
        self.user_book = self.loadUserBook()

    def loadUserData(self):
        user_mess = dict()
        for line in open(self.file_user,"r",encoding="ISO-8859-1"):
            if line.startswith("\"User-ID\""):
                continue
            if len(line.split(";")) != 3:
                continue
            line = line.strip().replace(" ","")
            userid, addr, age = [one.replace("\"","") for one in line.split(";")]
            if age == "NULL" or int(age) not in range(1,120):
                continue
            user_mess.setdefault(userid,{})
            user_mess[userid]["age"] = int(age)
            if len(addr.split(",")) < 3:
                continue
            city, province, country = addr.split(",")[-3:]
            user_mess[userid]["country"] = country
            user_mess[userid]["province"] = province
            user_mess[userid]["city"] = city
        return user_mess

    def loadBookMess(self):
        book_mess = dict()
        for line in open(self.file_book,"r",encoding="ISO-8859-1"):
            if line.startswith("\"ISBN\""):
                continue
            isbn, book_name = line.replace("\"","").split(";")[:2]
            book_mess[isbn] = book_name
        return book_mess

    def loadUserBook(self):
        user_book = dict()
        for line in open(self.file_rate,"r",encoding="ISO-8859-1"):
            if line.startswith("\"User-ID\""):
                continue
            user_id, isbn, rating = line.replace("\"","").split(";")[:3]
            user_book.setdefault(user_id,list())
            if int(rating) > 5:
                user_book[user_id].append(isbn)
        return user_book

    def show(self, X, Y, X_label, Y_label="数目"):
        plt.xlabel(X_label)
        plt.ylabel(Y_label)
        plt.xticks(np.arange(len(X)),X, rotation = 90)
        for a, b in zip(np.arange(len(X)),Y):
            plt.text(a, b, b, rotation = 45)
        plt.bar(np.arange(len(X)),Y)
        plt.show()

    def diffAge(self):
        age_user = dict()
        for key in self.user_mess.keys():
            age_split = int(int(self.user_mess[key]["age"]) / 10)
            age_user.setdefault(age_split,0)
            age_user[age_split] += 1
        age_user_sort = sorted(age_user.items(), key=lambda x:x[0], reverse=False)
        X = [x[0] for x in age_user_sort]
        Y = [x[1] for x in age_user_sort]
        print(age_user_sort)
        self.show(X,Y, X_label="用户年龄段")

    def diffpro(self):
        pro_user = dict()
        for key in self.user_mess.keys():
            if "province" in self.user_mess[key].keys() and self.user_mess[key]["province"] != "n/a":
                pro_user.setdefault(self.user_mess[key]["province"], 0)
                pro_user[self.user_mess[key]]["province"] += 1
            pro_user_sort = sorted(pro_user.items(), key=lambda x:x[1],reverse=True)[:20]
            X = [x[0] for x in pro_user_sort]
            Y = [x[1] for x in pro_user_sort]
            print(pro_user_sort)
            self.show(X, Y, X_label="用户所在州")

    def diffUserAge(self):
        age_books = dict()
        age_books.setdefault(1,dict())
        age_books.setdefault(2,dict())
        for key in self.user_mess.keys():
            if "country" not in self.user_mess[key].keys():
                continue
            if key not in self.user_book.keys():
                continue
            if int(self.user_mess[key]["age"]) in range(0,30):
                for book in self.user_book[key]:
                    if book not in self.book_mess.keys():
                        continue
                    age_books[1].setdefault(book,0)
                    age_books[1][book] += 1
            if int(self.user_mess[key]["age"]) in range(50,120):
                for book in self.user_book[key]:
                    if book not in self.book_mess.keys():
                        continue
                    age_books[2].setdefault(book,0)
                    age_books[2][book] += 1
        print("年龄在30岁以下的用户偏好共性图书top10")
        for one in sorted(age_books[1].items(), key=lambda x:x[1], reverse=True)[:10]:
            print(self.book_mess[one[0]])
        print("年龄在50岁以上的用户偏好共性图书top10")
        for one in sorted(age_books[2].items(), key=lambda x:x[1], reverse=True)[:10]:
            print(self.book_mess[one[0]])


if __name__ == "__main__":
    ushow = UserShow()
    ushow.diffUserAge()