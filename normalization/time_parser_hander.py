#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/25 14:52
# @Author  : honeyding
# @File    : time_parser_hander.py
# @Software: PyCharm

import time_slot_value
import pandas as pd
from common import entity


class TimeParserHander:

    def doParse(self,c_entity):
        timeSlotValue = time_slot_value.TimeSlotValue(c_entity, '2019-01-22 12:00:00')
        time = timeSlotValue.parse()
        return time


if __name__ == '__main__':
    entity = entity.entity()
    entity.entity = u'正月初三'
    entity.type = 'time_point'
    entity.abs_rel = 'relative'
    entity.is_refer = False
    entity.is_freq = False
    timeSlotValue = time_slot_value.TimeSlotValue(entity, '2019-01-22 12:00:00')
    time = timeSlotValue.parse()
    print("result:" + str(time.date) + '\t\n')
    print('start:' + str(time.startDate) + '\t\n')
    print('end:' + str(time.endDate) + '\t\n')
    print('lunar:' + str(time.lunar) + '\t\n')
    print('timeFormatInfo:' + time.timeFormatInfo + '\t\n')
    # file_list = [u'test/point-test.csv']
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
    #         basetime = df[u"basetime"]
    #         result = df[u"result"]
    #
    #         print (entity)
    #         ent = entity.entity()
    #         ent.entity = entity
    #         ent.type = "time_point"
    #         ent.abs_rel = abs_rel
    #         ent.is_refer = False
    #         ent.is_freq = False
    #         if u'上一天' in entity:
    #             print (entity)
    #         if pd.isnull(basetime):
    #             basetime = None
    #
    #         timeSlotValue = time_slot_value.TimeSlotValue(ent, basetime)
    #         time = timeSlotValue.parse()
    #
    #         if pd.isnull(result):
    #             if time:
    #                 print(entity + '\t\n')
    #                 print("result:" + time.date + '\t\n')
    #                 print('start:' + time.startDate + '\t\n')
    #                 print('end:' + time.endDate + '\t\n')
    #         elif time:
    #             # if time.date != result:
    #                 print('entity:' + entity + '\t\n')
    #                 print('old:' + str(result) + '\t\n')
    #                 print('result:' + time.date + '\t\n')
    #                 print('start:' + time.startDate + '\t\n')
    #                 print('end:' + time.endDate + '\t\n')

