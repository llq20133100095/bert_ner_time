#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/23 22:41
# @Author  : honeyding
# @File    : lunarMonth.py
# @Software: PyCharm

import re
import arrow
from utils import util_tools,digitconv

class lunarDay:
    # relative_day_diviation_pat = re.compile(u'.*?(?<![\\d])初([一二三四五六七八九十]|一十|十[一二三四五六七八九十]|二十[一二三四五六七八九十]|二十|三十[一]?)(之前|之后|前|后).*?')
    absolute_day_pat = re.compile(u'.*?(?P<day>初([一二三四五六七八九十]|一十|(十|二十)[一二三四五六七八九十]|二十|三十[一]?)).*?')

    # def get_relative_day(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
    #     matcher = self.relative_day_diviation_pat.match(entity)
    #     util = util_tools.Util()
    #     time = arrow.get(basetime)
    #     base_day = time.day
    #     if matcher:
    #         if matcher.group(1):
    #             day_ = matcher.group(1)
    #             day = digitconv.getNumFromHan(day_)
    #             shift_day = util.rel_deviation_num(matcher.group(2), day)
    #             if commonParser:
    #                 commonParser.lunar = True
    #                 commonParser.timeUnit[7] = True
    #                 commonParser.date = commonParser.date.shift(days=shift_day)
    #             return day_ + base_day
    #     return None

    def get_absolute_day(self, entity, commonParser=None):
        matcher = self.absolute_day_pat.match(entity)
        if matcher:
            if matcher.group("day"):
                day_ = matcher.group("day")[1:]
                day = digitconv.getNumFromHan(day_)
                if commonParser:
                    commonParser.date = commonParser.date.replace(day=day)
                    commonParser.timeUnit[3] = True
                    commonParser.lunar = True
            return day_
        return None

    def get_day(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        # day = self.get_relative_day(entity, basetime, commonParser)
        # if day:
        #     return day
        return self.get_absolute_day(entity, commonParser)


if __name__ == '__main__':
    day_proc = lunarDay()
    assert day_proc.get_absolute_month(u'初三') == 3
