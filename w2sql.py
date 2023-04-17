import pymysql
# import datetime
# ---------连接--------------
class Writer:
    def __init__(self) -> None:
        self.connect = pymysql.connect(host='127.0.0.1',   # 本地数据库
                                user='root',
                                password='123456',
                                db='traceit',
                                charset='utf8') #服务器名,账户,密码，数据库名称
        self.cur = self.connect.cursor()
    def Write(self,table,msg):
        sql = "" 
        try:
            # dt=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cur.execute("insert into %s (xsize,ysize,band,name,path,message) \
                                                values(%d,%d,%d,'%s','%s','%s')" % \
                                                      (table,msg[0],msg[1],msg[2],msg[3],msg[4],msg[5]))
        except Exception as e:
            print("插入数据失败:", e)
        else:
            # 如果是插入数据， 一定要提交
            self.connect.commit()

    def Close(self):
        self.cur.close()
        self.connect.close()

if __name__ == "__main__":
    wt = Writer()
    wt.Write('entry',(512,512,3,'abd','vs/pro.tif','WHUT'))
    wt.Close()