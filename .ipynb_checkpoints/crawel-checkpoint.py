# 图片来源于Pixabay: https://pixabay.com/zh/
import requests as req
import math
import re
from bs4 import BeautifulSoup

origin_img_link = []

# 获取page页有关key图片信息
def get_picture_page(key,page):
    i = 1
    while(i <= page):
        print('------------这是第%d页----------' %i)
        origin_rul = 'https://pixabay.com/zh/images/search/'+ key +'/?pagi=' + str(i)
        r = req.get(origin_rul)
        bs = BeautifulSoup(r.content, 'html.parser') #解析网页
        hyperlink = bs.find_all(name = 'img')
        for h in hyperlink:
            hh = h.get('src')
            print(hh)
            origin_img_link.append(hh)
        i += 1

get_picture_page('美女',5)        

print('开始下载\n\n\n')

# 按匹配下载    
for m in origin_img_link:
    if(re.match(r'^((https|http|ftp|rtsp|mms)?:\/\/)[a-zA-z]+.[a-zA-z]+.[a-zA-z]+/photo/\d{4}/\d{2}/\d{2}/\d{2}/\d{2}/',m)):
        r = req.get(m)
        s = re.sub(r'^((https|http|ftp|rtsp|mms)?:\/\/)[a-zA-z]+.[a-zA-z]+.[a-zA-z]+/photo/\d{4}/\d{2}/\d{2}/\d{2}/\d{2}/','',m)  # 正则表达式把https://给替换掉
        if r.status_code == 200:
            with open(s, 'wb') as f:
                f.write(r.content)
            print(s + '下载成功！')

print('（…^&^）下载完成')            
