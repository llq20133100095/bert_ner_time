#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/23 22:41
# @Author  : honeyding
# @File    : lunarMonth.py
# @Software: PyCharm

import re
import arrow
from utils import util_tools,digitconv

class lunarMonth:
    relative_month_diviation_pat = re.compile(u'.*?(?<![\\d])(正月|腊月|冬月|大年|元月)(之前|之后|前|后).*?')
    absolute_month_pat = re.compile(u'.*?(正月|腊月|冬月|大年|元月)(?P<day>初[一二三四五六七八九十]|一十|(十|二十)[一二三四五六七八九十]|二十|三十[一]?)?.*?')

    def get_relative_month(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        matcher = self.relative_month_diviation_pat.match(entity)
        util = util_tools.Util()
        time = arrow.get(basetime)
        base_month = time.month
        if matcher:
            month_ = util.lunar_month_num(matcher.group(1))
            shift_month = util.rel_deviation_num(matcher.group(2), month_)
            if commonParser:
                commonParser.lunar = True
                commonParser.timeUnit[7] = True
                commonParser.date = commonParser.date.shift(months=shift_month)
            return month_ + base_month
        return None

    def get_absolute_month(self, entity, commonParser=None):
        matcher = self.absolute_month_pat.match(entity)
        util = util_tools.Util()
        if matcher:
            if matcher.group(1):
                month_ = util.lunar_month_num(matcher.group(1))
                if commonParser:
                    commonParser.lunar = True
                    commonParser.timeUnit[2] = True
                    commonParser.date = commonParser.date.replace(month=month_)
                if matcher.group("day"):
                    if matcher.group("day").startswith(u'初'):
                        day_ = matcher.group("day")[1:]
                    else:
                        day_ = matcher.group("day")
                    day = digitconv.getNumFromHan(day_)
                    if commonParser:
                        commonParser.date = commonParser.date.replace(day=day)
                        commonParser.timeUnit[3] = True
                return month_
        return None

    def get_month(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        month = self.get_relative_month(entity, basetime, commonParser)
        if month:
            return month
        return self.get_absolute_month(entity, commonParser)


if __name__ == '__main__':
    month_proc = lunarMonth()
    assert month_proc.get_absolute_month(u'正月初三') == 1
    assert month_proc.get_absolute_month(u'腊月') == 11
    assert month_proc.get_absolute_month(u'大年') == 1