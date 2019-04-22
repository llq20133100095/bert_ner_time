#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: charlielai 
# time:2018/11/22


"""
download langconv.py and zh_wiki.py
https://raw.githubusercontent.com/skydark/nstools/master/zhtools/langconv.py 
https://raw.githubusercontent.com/skydark/nstools/master/zhtools/zh_wiki.py
put these 2 files in the same dir with code 
zh_wiki.py:https://github.com/skydark/nstools/blob/master/zhtools/zh_wiki.py
langconv.py:https://github.com/skydark/nstools/blob/master/zhtools/langconv.py

"""


from langconv import *


def tradition2simple(text):
    """  convert tradition to simple
    :param text: 
    :return text: 
    """
    text = Converter('zh-hans').convert(text)
    # text = Converter('zh-hans').convert(text.decode('utf-8'))
    # text = text.encode('utf-8')
    return text


CN_NUM = {u'零': 0,
          u'壹': 1,
          u'贰': 2,
          u'叁': 3,
          u'肆': 4,
          u'伍': 5,
          u'陆': 6,
          u'柒': 7,
          u'捌': 8,
          u'玖': 9,
          u'〇': 0,
          u'一': 1,
          u'二': 2,
          u'三': 3,
          u'四': 4,
          u'五': 5,
          u'六': 6,
          u'七': 7,
          u'八': 8,
          u'九': 9,
          u'貮': 2,
          u'两': 2,
          u'十': 10,
          u'百': 100,
          u'千': 1000,
          u'万': 10000,
          u'亿': 100000000,
          u'○': 0,
          u'0': 0,
          u'１': 1,
          u'３': 3,
          u'５': 5,
          u'９': 9,
          u'1': 1,
          u'2': 2,
          u'3': 3,
          u'4': 4,
          u'5': 5,
          u'6': 6,
          u'7': 7,
          u'8': 8,
          u'9': 9,
          u'壹': 1,
          u'贰': 2,
          u'叁': 3,
          u'肆': 4,
          u'廿': 2,
          u'天': 7,
          u'日': 7,
          }


def getNumFromHan(text):
    """ convert chinese date to digit format
    :param text: 
    :return: text
    """
    text = text.strip()
    if len(text) !=0 and isUnit(text[0]):
        text = u"一" + text

    hasUnit = False
    for arr in text:
        if isUnit(arr):
            hasUnit = True

    if hasUnit:
        tmp = 0
        resList = []
        for res,arr in enumerate(text):
            if not isUnit(arr):
                tmp += CN_NUM.get(arr)
                if res == len(text)-1:
                    resList.append(tmp)

            else:
                tmp *= CN_NUM.get(arr)
                if res < len(text) - 1 and not isUnit(text[res+1]) or res == len(text)-1:
                    resList.append(tmp)
                    tmp = 0


        res = 0
        for var8 in resList:
            res += var8
        return res
    else:
        sb = []
        for arr in text:
            sb.append(str(CN_NUM.get(arr)))

    return int(''.join(sb))



def isUnit(c):
    isUnit = False
    if c in [u'十', u'百', u'千', u'万', u'亿']:
        isUnit = True
    return isUnit

if __name__ == '__main__':
    # simplified_sentence = '憂郁的臺灣烏龜'
    # print(tradition2simple(simplified_sentence))
    chinese_date = u'二零17'
    print(getNumFromHan(chinese_date))


# test()