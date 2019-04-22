#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/01 23:00
# @Author  : honeyding
# @File    : minute.py
# @Software: PyCharm

import re
import arrow
from utils import util_tools,digitconv
import pandas as pd

class Minute:
    relative_minute_pat = re.compile(u'.*?(上|下){0,}(本|上+|下+|这)(一?个?)分钟?.*?')
    absolute_minute_pat = re.compile(u'.*?([\\d]+)分钟?(许)?.*?')
    ke_pat = re.compile(".*?(?P<minute>[一二三123])刻.*?")

    def get_relative_minute(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        matcher = self.relative_minute_pat.match(entity)
        decorate_deviation,deviation = '',''
        if matcher:
            if matcher.group(1):
                decorate_deviation = matcher.group(1)
            if matcher.group(2):
                deviation = matcher.group(2)
            time = arrow.get(basetime)
            if deviation != '':
                util = util_tools.Util()
                realMinute = time.minute + util.rel_deviation(deviation, decorate_deviation)
                if commonParser:
                    commonParser.timeUnit[7] = True
                return realMinute
        return None

    def get_absolute_minute(self, entity, commonParser=None):
        matcher = self.absolute_minute_pat.match(entity)
        ke_matcher = self.ke_pat.match(entity)
        if matcher:
            minute_ = digitconv.getNumFromHan(matcher.group(1))
            if commonParser:
                if matcher.group(2):
                    commonParser.timeUnit[7] = True
                else:
                    commonParser.timeUnit[6] = True
                if minute_ == 60:
                    commonParser.date = commonParser.date.replace(hour=commonParser.date.hour+1).replace(minute=0)
                elif 0 <= minute_ < 60:
                    commonParser.date = commonParser.date.replace(minute=minute_)
                    return None
                else:
                    commonParser.timeFormatInfo = u'时间表述异常，分钟应该在[0,60]范围内！'
                    return None
            return minute_
        elif ke_matcher:
            minuteToParse = digitconv.getNumFromHan(ke_matcher.group("minute"))
            minute_ = minuteToParse * 15
            if commonParser:
                if matcher.group(2):
                    commonParser.timeUnit[7] = True
                else:
                    commonParser.timeUnit[6] = True
                if minute_ == 60:
                    commonParser.date = commonParser.date.replace(hour=commonParser.date.hour + 1).replace(minute=0)
                elif 0 <= minute_ < 60:
                    commonParser.date = commonParser.date.replace(minute=minute_)
                    return None
                else:
                    commonParser.timeFormatInfo = u'时间表述异常，分钟应该在[0,60]范围内！'
                    return None
            return minute_
        return None

    def get_minute(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        minute = self.get_relative_minute(entity, basetime, commonParser)
        if not minute is None:
            return minute
        return self.get_absolute_minute(entity, commonParser)

if __name__ == '__main__':
    minute_proc = Minute()
    assert minute_proc.get_minute(u'2007年3月21号') is None
    assert minute_proc.get_minute(u'下一分钟') == 1
    assert minute_proc.get_minute(u'这分钟') == 0

    assert minute_proc.get_minute(u'30分钟') == 30

    file_list = [u'../test/minute_testcase.csv']

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
                    if type == u'time_point':
                        if pd.isnull(abs_rel):
                            continue
                        else:
                            day = None
                            print (entity)
                            if type == u'time_point':
                                if abs_rel == u'relative':
                                    day = minute_proc.get_minute(entity, basetime)
                                elif abs_rel == u'absolute':
                                    day = minute_proc.get_minute(entity)

                            if pd.isnull(basetime):
                                basetime = ''
                            if pd.isnull(result) and day is None:
                                continue
                            elif day is None:
                                print(
                                        'entity:' + entity + ' \t ' + 'type:' + type + ' \t ' + 'abs_rel:' + abs_rel + ' \t ' + 'basetime: ' + basetime + '\t\n')
                                print('result:' + str(result) + ' \t ' + 'ret_result: None\t\n')
                            elif pd.isnull(result):
                                print(
                                        'entity:' + entity + ' \t ' + 'type:' + type + ' \t ' + 'abs_rel:' + abs_rel + ' \t ' + 'basetime: ' + basetime + '\t\n')
                                print('result:None \t ' + 'ret_result:' + str(day) + '\t\n')
                            elif day != int(result):
                                print(
                                        'entity:' + entity + ' \t ' + 'type:' + type + ' \t ' + 'abs_rel:' + abs_rel + ' \t ' + 'basetime: ' + basetime + '\t\n')
                                print('result:' + str(result) + ' \t ' + 'ret_result:' + str(day) + '\t\n')