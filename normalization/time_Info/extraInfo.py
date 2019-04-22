#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/24 16:02
# @Author  : honeyding
# @File    : day.py
# @Software: PyCharm


import re
import arrow
from utils import util_tools

class ExtraInfo:
    #不能是第，应该重复匹配
    duration_pat = re.compile(u'(\d)+个?(世纪|年|月|季度|星期|周|天|小时|分钟?|秒钟?)')

    def get_duration_info(self, entity):
        matcher = self.duration_pat.match(entity)
        decorate_deviation, deviation = '', ''
        if matcher:
            if matcher.group(1):
                decorate_deviation = matcher.group(1)
            if matcher.group(2):
                deviation = matcher.group(2)

                # duration[0] = True
                # return day
        return None


if __name__ == '__main__':
    info_proc = ExtraInfo()
    # assert info_proc.get_duration_info(u'3天') is None
    # assert info_proc.get_relative_day(u'12年') == 2

    # assert day_proc.get_absolute_day(u'2007年3月21号') == 21