#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/25 11:05
# @Author  : honeyding
# @File    : TimeSlotValue.py
# @Software: PyCharm

import arrow
import CommonParser
from time_Info.year import Year
from time_Info.month import Month
from time_Info.day import Day
from time_Info.apm import Apm
from time_Info.hour import Hour
from time_Info.minute import Minute
from time_Info.second import Second
import calendar

class TimeSlotValue:
    def __init__(self,timelist,basetime=None):
        self.timelist = timelist
        self.timeScaleRecord = []
        self.timeUnit = None
        self.timeAPMInfo = None
        self.basetime = basetime
        self.formatText = None

    def toFormatText(self,date):
        return arrow.get(date, 'YYYY-MM-DD HH:mm:ss')

    def parse(self):

        if self.timelist.type == 'time_point':
            if self.timelist.abs_rel == 'relative':
                commonParser = CommonParser.CommonParser(self.basetime)
            else:
                commonParser = CommonParser.CommonParser()
            if not commonParser.timeUnit[1]:
                year_proc = Year()
                year_proc.get_year(self.timelist.entity, self.basetime, commonParser)
            if not commonParser.timeUnit[2]:
                month_proc = Month()
                month_proc.get_month(self.timelist.entity, self.basetime, commonParser)
            if not commonParser.timeUnit[3]:
                day_proc = Day()
                day_proc.get_day(self.timelist.entity, self.basetime, commonParser)
            if not commonParser.timeUnit[4]:
                apm_proc = Apm()
                apm_proc.get_apm_info(self.timelist.entity, commonParser)
            if not commonParser.timeUnit[5]:
                hour_proc = Hour()
                hour_proc.get_hour(self.timelist.entity, self.basetime, commonParser)
            if not commonParser.timeUnit[6]:
                minute_proc = Minute()
                minute_proc.get_minute(self.timelist.entity, self.basetime, commonParser)
            if not commonParser.timeUnit[7]:
                second_proc = Second()
                second_proc.get_second(self.timelist.entity, self.basetime, commonParser)

            if commonParser.timeUnit[7]:
                commonParser.date = commonParser.date.format('YYYY-MM-DD HH:mm:ss')
                commonParser.startDate = commonParser.date
                commonParser.endDate = commonParser.date
            elif commonParser.timeUnit[6]:
                commonParser.startDate = commonParser.date.replace(second=0).format('YYYY-MM-DD HH:mm:ss')
                commonParser.endDate = commonParser.date.replace(second=59).format('YYYY-MM-DD HH:mm:ss')
            elif commonParser.timeUnit[5]:
                commonParser.startDate = commonParser.date.replace(minute=0).replace(second=0).format('YYYY-MM-DD HH:mm:ss')
                commonParser.endDate = commonParser.date.replace(minute=59).replace(second=59).format('YYYY-MM-DD HH:mm:ss')
            elif commonParser.timeUnit[3]:
                commonParser.startDate = commonParser.date.replace(hour=0).replace(minute=0).replace(second=0).format(
                    'YYYY-MM-DD HH:mm:ss')
                commonParser.endDate = commonParser.date.replace(hour=23).replace(minute=59).replace(second=59).format(
                    'YYYY-MM-DD HH:mm:ss')
            elif commonParser.timeUnit[2]:
                monthRange = calendar.monthrange(commonParser.date.year, commonParser.date.month)
                commonParser.startDate = commonParser.date.replace(day=1).replace(hour=0).replace(minute=0).replace(second=0).format(
                    'YYYY-MM-DD HH:mm:ss')
                commonParser.endDate = commonParser.date.replace(day=monthRange[1]).replace(hour=23).replace(minute=59).replace(second=59).format(
                    'YYYY-MM-DD HH:mm:ss')
            elif commonParser.timeUnit[1]:
                commonParser.startDate = commonParser.date.replace(month=1).replace(day=1).replace(hour=0).replace(minute=0).replace(second=0).format('YYYY-MM-DD HH:mm:ss')
                monthRange = calendar.monthrange(commonParser.date.year, 12)
                commonParser.endDate = commonParser.date.replace(month=12).replace(day=monthRange[1]).replace(hour=23).replace(minute=59).replace(second=59).format('YYYY-MM-DD HH:mm:ss')
            else:
                return None
            if u'公元前' in self.timelist.entity:
                commonParser.startDate = '-'+commonParser.startDate
                commonParser.endDate = '-'+commonParser.endDate

            commonParser.date = commonParser.startDate
            return commonParser
