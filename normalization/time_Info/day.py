#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/24 16:02
# @Author  : honeyding
# @File    : day.py
# @Software: PyCharm


import re
import arrow
from utils import util_tools, digitconv
import pandas as pd

class Day:
    relative_day_pat = re.compile(u'.*?(大+)?(前|昨|今|当|明|次|后)[天日].*?')
    week_day_pat = re.compile(u'.*?([下上]+)?个?(星期|礼拜|周)([1-7一二三四五六日天]).*?')
    absolute_day_pat = re.compile(u'.*?([\\d一二三四五六七八九]+)[日号].*?')

    def get_relative_day(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        matcher = self.relative_day_pat.match(entity)
        decorate_deviation, deviation = '', ''
        time = arrow.get(basetime)
        util = util_tools.Util()
        base_day = time.day
        if matcher:
            if matcher.group(1):
                decorate_deviation = matcher.group(1)
            shift_day = util.rel_deviation(matcher.group(2), decorate_deviation)
            if commonParser:
                commonParser.timeUnit[3] = True
                commonParser.date = commonParser.date.shift(days=shift_day)
            return shift_day
        else:
            weekday = time.weekday()#basetime是周四
            matcher = self.week_day_pat.match(entity)
            if matcher:
                week_day = util.week_deviation_num(matcher.group(1), matcher.group(3))
                shift_day = week_day - weekday
                if commonParser:
                    commonParser.timeUnit[3] = True
                    commonParser.date = commonParser.date.shift(days=shift_day)
                return shift_day

    def get_absolute_day(self, entity, commonParser=None):
        matcher = self.absolute_day_pat.match(entity)
        if matcher:
            if matcher.group(1):
                day_ = digitconv.getNumFromHan(matcher.group(1))
                if commonParser:
                    commonParser.timeUnit[3] = True
                    if 1 <= day_ <= 31:
                        commonParser.date = commonParser.date.replace(day=day_)
                    else:
                        commonParser.timeFormatInfo = u'时间表述异常，日应该在[1,31]范围内！'
                        return None
                return day_
        return None

    def get_day(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        day = self.get_absolute_day(entity, commonParser)
        if not day is None:
            return day
        return self.get_relative_day(entity, basetime, commonParser)


if __name__ == '__main__':
    day_proc = Day()
    assert day_proc.get_relative_day(u'2007年3月21号') is None
    assert day_proc.get_relative_day(u'明天') == 1
    assert day_proc.get_relative_day(u'今天') == 0
    assert day_proc.get_relative_day(u'周日') == 3

    file_list = [u'../test/day_testcase.csv']

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

            if pd.isnull(entity):
                continue
            else:
                if pd.isnull(type):
                    continue
                else:
                    if type == u'time_point' or type == u'time_period':
                        if pd.isnull(abs_rel):
                            continue
                        else:
                            day = None
                            print (entity)
                            if type == u'time_point':
                                if abs_rel == u'relative':
                                    day = day_proc.get_day(entity, basetime)
                                elif abs_rel == u'absolute':
                                    day = day_proc.get_absolute_day(entity)
                            else:
                                # print (entity)
                                day = day_proc.get_day(entity)

                            if pd.isnull(basetime):
                                basetime = ''
                            if pd.isnull(result) and day is None:
                                continue
                            elif day is None:
                                print('entity:' + entity + ' \t ' + 'type:' + type + ' \t ' + 'abs_rel:' + abs_rel + ' \t ' + 'basetime: ' + basetime + '\t\n')
                                print('result:' + str(result) + ' \t ' + 'ret_result: None\t\n')
                            elif pd.isnull(result):
                                print('entity:' + entity + ' \t ' + 'type:' + type + ' \t ' + 'abs_rel:' + abs_rel + ' \t ' + 'basetime: ' + basetime + '\t\n')
                                print('result:None \t ' + 'ret_result:' + str(day) + '\t\n')
                            elif day != int(result):
                                print('entity:' + entity + ' \t ' + 'type:' + type + ' \t ' + 'abs_rel:' + abs_rel + ' \t ' + 'basetime: ' + basetime + '\t\n')
                                print('result:' + str(result) + ' \t ' + 'ret_result:' + str(day) + '\t\n')
