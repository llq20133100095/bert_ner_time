#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/25 14:52
# @Author  : honeyding
# @File    : TimeParserHander.py
# @Software: PyCharm

import TimeSlotValue
from common import entity

import pandas as pd

class TimeParserHander:

    def doParse(self):
        ent = entity.entity()
        ent.entity = u"1998年3月21日"
        ent.start = 0
        ent.end = 10
        ent.type = "time_point"
        ent.abs_rel = "absolute"
        ent.is_refer = False
        ent.is_freq = False
        timeSlotValue = TimeSlotValue.TimeSlotValue(ent,'2018-11-01 12:00:00')
        time = timeSlotValue.parse()


if __name__ == '__main__':
    TimeParserHander().doParse()
    file_list = [u'test/point_relative.csv']

    indexs, sentences, entities = [], [], []
    for file in file_list:
        f = pd.read_csv(file, encoding=u'utf-8')
        sen = ''
        _index = None
        for i, (idx, df) in enumerate(f.iterrows()):
            entity = df[u"entity"]
            type = df[u"type"]
            abs_rel = df[u"abs_rel"]
            basetime = df[u"basetime"]
            result = df[u"result"]

            print (entity)
            ent = Entity.entity()
            ent.entity = entity
            ent.type = "time_point"
            ent.abs_rel = abs_rel
            ent.is_refer = False
            ent.is_freq = False
            if u'晚上12点' in entity:
                print (entity)
            timeSlotValue = TimeSlotValue.TimeSlotValue(ent,basetime)
            time = timeSlotValue.parse()

            if pd.isnull(result):
                if time:
                    print(entity + '\t\n')
                    print("result:" + time.date + '\t\n')
                    print('start:' + time.startDate + '\t\n')
                    print('end:' + time.endDate + '\t\n')
            elif time:
                # if time.date != result:
                    print('entity:' + entity + '\t\n')
                    print('old:' + str(result) + '\t\n')
                    print('result:' + time.date + '\t\n')
                    print('start:' + time.startDate + '\t\n')
                    print('end:' + time.endDate + '\t\n')

    # file_list = [u'test/point_absolute.csv']
    #
    # indexs, sentences, entities = [], [], []
    # for file in file_list:
    #     f = pd.read_csv(file, encoding=u'utf-8')
    #     sen = ''
    #     _index = None
    #     for i, (idx, df) in enumerate(f.iterrows()):
    #         entity = df[u"entity"]
    #         type = df[u"type"]
    #         abs_rel = df[u"abs_rel"]
    #         result = df[u"result"]
    #
    #         print (entity)
    #         ent = Entity.entity()
    #         ent.entity = entity
    #         ent.type = "time_point"
    #         ent.abs_rel = "absolute"
    #         ent.is_refer = False
    #         ent.is_freq = False
    #         timeSlotValue = TimeSlotValue.TimeSlotValue(ent)
    #         if u'2014年1月1日0时0分38秒' in entity:
    #             print (entity)
    #         time = timeSlotValue.parse()
    #
    #         if pd.isnull(result):
    #             if time:
    #                 print(entity + '\t\n')
    #                 print("result:"+time.date+'\t\n')
    #         elif time:
    #             if time.date != result:
    #                 print('entity:' + entity+ '\t\n')
    #                 print('old:' + str(result) + '\t\n')
    #                 print('result:' + time.date + '\t\n')

