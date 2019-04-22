#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/01 23:05
# @Author  : honeyding
# @File    : Second.py
# @Software: PyCharm

import re
import arrow
from utils import util_tools,digitconv
import pandas as pd

class Second:
    relative_second_pat = re.compile(u'.*?(上|下){0,}(本|上+|下+|这)(一?个?)秒钟?.*?')
    absolute_second_pat = re.compile(u'.*?(?<![\\d零一二三四五六七八九十])([\\d零一二三四五六七八九十]+)秒钟?.*?')

    def get_relative_second(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        matcher = self.relative_second_pat.match(entity)
        decorate_deviation,deviation = '',''
        if matcher:
            if matcher.group(1):
                decorate_deviation = matcher.group(1)
            if matcher.group(2):
                deviation = matcher.group(2)
            time = arrow.get(basetime)
            if deviation != '':
                util = util_tools.Util()
                realSecond = time.second + util.rel_deviation(deviation, decorate_deviation)
                if commonParser:
                    commonParser.timeUnit[7] = True
                return realSecond
        return None

    def get_absolute_second(self, entity, commonParser=None):
        matcher = self.absolute_second_pat.match(entity)
        if matcher:
            chsecond = matcher.group(1)
            second_ = digitconv.getNumFromHan(chsecond)
            if commonParser:
                commonParser.timeUnit[7] = True
                commonParser.date = commonParser.date.replace(second=second_)
            return second_
        return None

    def get_second(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        second = self.get_relative_second(entity, basetime, commonParser)
        if not second is None:
            return second
        return self.get_absolute_second(entity, commonParser)

if __name__ == '__main__':
    second_proc = Second()
    assert second_proc.get_second(u'2007年3月21号') is None
    assert second_proc.get_second(u'下一秒钟') == 1
    assert second_proc.get_second(u'这秒钟') == 0

    assert second_proc.get_second(u'30秒钟') == 30

    file_list = [u'../test/second_testcase.csv']

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
                                    day = second_proc.get_second(entity, basetime)
                                elif abs_rel == u'absolute':
                                    day = second_proc.get_second(entity)

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