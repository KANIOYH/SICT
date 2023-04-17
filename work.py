import API
import poster
import sys
import w2sql
if __name__ == "__main__":
    seq = int(sys.argv[1])
    path = sys.argv[2]
    out = sys.argv[3]
    name = sys.argv[4]
    msg = sys.argv[5]
    poster.client()
    if seq>0:
        x,y,band = API.embed(path,out,msg)
        wt = w2sql.Writer()
        path.replace('/',"//")
        wt.Write('entry',(x,y,band,name,path,msg))
        wt.Close()
        poster.send_ok(str(seq))
    else:
        rets=API.extract(path)
        res = "IPv4: " +rets[0]+"\nTime: "+rets[1]+"\nMessage: "+rets[2]
        poster.send_ok(res)
    poster.close()
    exit(996)