import API

if __name__ == "__main__":
    fileName = r'E:\test\LT001_out.TIF'   #原始文件
    path = r'E:\test\LT001_out_cut.TIF'        #生成的文件
    #SICT_API.embed(fileName, path, 'hello!')
    res = API.extract(path)
    print(res)