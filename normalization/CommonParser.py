#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/25 11:22
# @Author  : honeyding
# @File    : CommonParser.py
# @Software: PyCharm

import arrow

class CommonParser:
    def __init__(self,basttime=None):
        if basttime:
            self.date = arrow.get(basttime)
        else:
            self.date = arrow.utcnow().to('local')
        self.isSet = False
        self.TimeSpan = None
        self.timeScaleRecord = [False,False,False,False,False,False,False,False]
        #世纪、年、月、日、apm、时、分、秒
        self.timeUnit = [False,False,False,False,False,False,False,False]
        #年、月、周、日、时、分、秒: time_frequency and time_duration
        self.timeLength = [False,False,False,False,False,False,False]
        self.timeAPMInfo = None
        self.TimeScaleNum = 8
        self.startDate = arrow.utcnow().to('local')
        self.endDate = arrow.utcnow().to('local')
        #is approximation
        self.is_approximation = False
        self.timeFormatInfo = ""
        self.lunar = False

    def setDate(self, date, timeScaleRecord, timeUnit, timeAPMInfo):
        self.date = date
        self.timeScaleRecord = timeScaleRecord
        self.timeUnit = timeUnit
        self.timeAPMInfo = timeAPMInfo
        self.date.replace(second=0)

        for i in range(len(timeScaleRecord)-2, -1, -1):
            if timeScaleRecord[i]:
                break
            if i == 5:
                self.date.replace(minute=0)
            elif i == 4:
                self.date.replace(hour=0)
