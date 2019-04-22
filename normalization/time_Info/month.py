#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/01 22:41
# @Author  : honeyding
# @File    : month.py
# @Software: PyCharm

import re
import arrow
from utils import util_tools,digitconv
import pandas as pd

class Month:
    relative_month_pat = re.compile(u'.*?(上|下){0,}(本|上|下|这|那)(一?个?)月.*?')
    relative_month_diviation_pat = re.compile(u'.*?([0-9零一二三四五六七八九十两])个月(之前|之后|前|后).*?')
    absolute_month_pat = re.compile(u'.*?([\\d零两一二三四五六七八九十]+)月.*?')

    def get_relative_month(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        matcher = self.relative_month_pat.match(entity)
        decorate_deviation,deviation = '',''
        util = util_tools.Util()
        time = arrow.get(basetime)
        base_month = time.month
        if matcher:
            if matcher.group(1):
                decorate_deviation = matcher.group(1)
            shift_month = util.rel_deviation(matcher.group(2), decorate_deviation)
            if commonParser:
                commonParser.timeUnit[2] = True
                if shift_month != 0:
                    commonParser.date = commonParser.date.shift(months=shift_month)
            return base_month + shift_month
        else:
            matcher = self.relative_month_diviation_pat.match(entity)
            if matcher:
                month_ = digitconv.getNumFromHan(matcher.group(1))
                shift_month = util.rel_deviation_num(matcher.group(2), month_)
                if commonParser:
                    commonParser.timeUnit[7] = True
                    commonParser.date = commonParser.date.shift(months=shift_month)
                return shift_month + base_month
        return None

    def get_absolute_month(self, entity, commonParser=None):
        matcher = self.absolute_month_pat.match(entity)
        if matcher:
            if matcher.group(1):
                month_ = digitconv.getNumFromHan(matcher.group(1))
                if commonParser:
                    commonParser.timeUnit[2] = True
                    if 1 <= month_ <= 12:
                        commonParser.date = commonParser.date.replace(month=month_)
                    else:
                        commonParser.timeFormatInfo = u'时间表述异常，月份应该在[1,12]范围内！'
                        return None
                return month_
        return None

    def get_month(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        month = self.get_relative_month(entity, basetime, commonParser)
        if not month is None:
            return month
        return self.get_absolute_month(entity, commonParser)

if __name__ == '__main__':
    month_proc = Month()
    assert month_proc.get_absolute_month(u'1992年2月') == 2
    assert month_proc.get_relative_month(u'2007年3月21号') is None
    assert month_proc.get_relative_month(u'3个月前') == 8
    assert month_proc.get_relative_month(u'这个月') == 11

    assert month_proc.get_absolute_month(u'2007年3月21号') == 3

    file_list = [u'../test/month_testcase.csv']

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
                            month = None
                            print (entity)
                            if u'十月初五' in entity:
                                continue
                            if type == u'time_point':
                                if abs_rel == u'relative':
                                    month = month_proc.get_month(entity, basetime)
                                elif abs_rel == u'absolute':
                                    month = month_proc.get_absolute_month(entity)
                            else:
                                # print (entity)
                                month = month_proc.get_month(entity)

                            if pd.isnull(basetime):
                                basetime = ''
                            if pd.isnull(result) and month is None:
                                continue
                            elif month is None:
                                print(
                                            'entity:' + entity + ' \t ' + 'type:' + type + ' \t ' + 'abs_rel:' + abs_rel + ' \t ' + 'basetime: ' + basetime + '\t\n')
                                print('result:' + str(result) + ' \t ' + 'ret_result: None\t\n')
                            elif pd.isnull(result):
                                print(
                                            'entity:' + entity + ' \t ' + 'type:' + type + ' \t ' + 'abs_rel:' + abs_rel + ' \t ' + 'basetime: ' + basetime + '\t\n')
                                print('result:None \t ' + 'ret_result:' + str(month) + '\t\n')
                            elif month != int(result):
                                print(
                                            'entity:' + entity + ' \t ' + 'type:' + type + ' \t ' + 'abs_rel:' + abs_rel + ' \t ' + 'basetime: ' + basetime + '\t\n')
                                print('result:' + str(result) + ' \t ' + 'ret_result:' + str(month) + '\t\n')