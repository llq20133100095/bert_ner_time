#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 13:24
# @Author  : honeyding
# @File    : year.py
# @Software: PyCharm

import re
import arrow
from utils import util_tools,digitconv
import pandas as pd

class Year:
    relative_year_pat = re.compile(u'.*?(大|上|下){0,}(前|去|上|今|明|后|来|同|次|翌|当)年.*?')
    relative_year_diviation_pat = re.compile(u'.*?([0-9零一二三四五六七八九十两]+)?年(之前|之后|前|后).*?')
    absolute_year_pat = re.compile(u'.*?([0-9○零一二三四五六七八九十１３５９]{2,4})财?年.*?')
    mix_year_pat = re.compile(u'.*?(?P<year>[\\d零一二三四五六七八九十]{4})([-/\\.](?P<month>(1[012]|0?[1-9]|[一二三四五六七八九十]|十[一二]))([-/\\.](?P<day>([12][0-9])|(3[01])|(0?[1-9])|(一?十[一二三四五六七八九]?)|(二十[一二三四五六七八九]?)|(三十一?)))?)?.*?')

    def get_relative_year(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        time = arrow.get(basetime)
        base_year = time.year
        util = util_tools.Util()
        matcher = self.relative_year_pat.match(entity)
        decorate_deviation = ''
        if matcher:
            if matcher.group(1):
                decorate_deviation = matcher.group(1)
            year_shift = util.rel_deviation(matcher.group(2), decorate_deviation)
            if commonParser:
                commonParser.timeUnit[1] = True
                commonParser.date = commonParser.date.shift(years=year_shift)
            return year_shift + base_year
        else:
            matcher = self.relative_year_diviation_pat.match(entity)
            if matcher:
                year_ = 1
                if matcher.group(1):
                    year_ = digitconv.getNumFromHan(matcher.group(1))
                year_shift = util.rel_deviation_num(matcher.group(2), year_)
                if commonParser:
                    commonParser.timeUnit[7] = True
                    commonParser.date = commonParser.date.shift(years=year_shift)
                return base_year + year_shift

        return None

    def get_absolute_year(self, entity, commonParser=None):
        matcher = self.absolute_year_pat.match(entity)
        if matcher:
            if matcher.group(1):
                year_ = digitconv.getNumFromHan(matcher.group(1))
                if commonParser:
                    if year_ < 100:
                        year_ple_tw = 2000 + year_
                        year_ple_ni = 1900 + year_
                        if abs(commonParser.date.year - year_ple_tw) >= abs(commonParser.date.year - year_ple_ni):
                            commonParser.date = commonParser.date.replace(year=year_ple_ni)
                        else:
                            commonParser.date = commonParser.date.replace(year=year_ple_tw)
                    else:
                        commonParser.date = commonParser.date.replace(year=year_)
                    commonParser.timeUnit[1] = True
                return year_
        else:
            matcher = self.mix_year_pat.match(entity)
            if matcher:
                year_ = digitconv.getNumFromHan(matcher.group("year"))
                if commonParser:
                    if year_ < 100:
                        year_ple_tw = 2000 + year_
                        year_ple_ni = 1900 + year_
                        if abs(commonParser.date.year - year_ple_tw) >= abs(commonParser.date.year - year_ple_ni):
                            commonParser.date = commonParser.date.replace(year=year_ple_ni)
                        else:
                            commonParser.date = commonParser.date.replace(year=year_ple_tw)
                    else:
                        commonParser.date = commonParser.date.replace(year=year_)
                    commonParser.timeUnit[1] = True
                    if matcher.group("month"):
                        month = digitconv.getNumFromHan(matcher.group("month"))
                        commonParser.date = commonParser.date.replace(month=month)
                        commonParser.timeUnit[2] = True
                        if matcher.group("day"):
                            day = digitconv.getNumFromHan(matcher.group("day"))
                            commonParser.date = commonParser.date.replace(day=day)
                            commonParser.timeUnit[3] = True
                return year_
        return None

    def get_year(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        year = self.get_relative_year(entity, basetime, commonParser)
        if not year is None:
            return year
        return self.get_absolute_year(entity, commonParser)

if __name__ == '__main__':
    year_proc = Year()
    assert year_proc.get_relative_year(u'从去年12月底') == 2017
    assert year_proc.get_relative_year(u'2007年3月21号') is None
    assert year_proc.get_relative_year(u'明年') == 2019
    assert year_proc.get_relative_year(u'今年') == 2018
    assert year_proc.get_relative_year(u'2年后') == 2020

    assert year_proc.get_absolute_year(u'2007.03') == 2007
    assert year_proc.get_absolute_year(u'2007年') == 2007
    assert year_proc.get_absolute_year(u'从2007年开始') == 2007


    file_list = [u'../test/year_testcase.csv']

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
                                year = None
                                print (entity)
                                if type == u'time_point':
                                    if abs_rel == u'relative':
                                        year = year_proc.get_relative_year(entity,basetime)
                                    elif abs_rel == u'absolute':
                                        year = year_proc.get_absolute_year(entity)
                                else:
                                    # print (entity)
                                    year = year_proc.get_year(entity)

                                if pd.isnull(basetime):
                                    basetime = ''
                                if pd.isnull(result) and year is None:
                                    continue
                                elif year is None:
                                    print('entity:' + entity + ' \t ' + 'type:' + type + ' \t ' + 'abs_rel:' + abs_rel + ' \t ' + 'basetime: ' + basetime + '\t\n')
                                    print('result:' + str(result) + ' \t ' + 'ret_result: None\t\n')
                                elif pd.isnull(result):
                                    print('entity:' + entity + ' \t ' + 'type:' + type + ' \t ' + 'abs_rel:' + abs_rel + ' \t ' + 'basetime: ' + basetime + '\t\n')
                                    print('result:None \t ' + 'ret_result:' + str(year) + '\t\n')
                                elif year != int(result):
                                    print('entity:' + entity + ' \t ' + 'type:' + type + ' \t ' + 'abs_rel:' + abs_rel + ' \t ' + 'basetime: ' + basetime + '\t\n')
                                    print('result:' + str(result) + ' \t ' + 'ret_result:' + str(year) + '\t\n')
