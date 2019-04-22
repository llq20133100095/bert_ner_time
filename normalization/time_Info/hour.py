#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/01 22:50
# @Author  : honeyding
# @File    : hour.py
# @Software: PyCharm

import re
import arrow
from utils import util_tools,digitconv
from normalization.time_Info.apm import Apm
import pandas as pd

class Hour:
    hour_minute_guo_pat = re.compile(u".*?([01]?\\d|2[0-3]|([一二三四五六七八九两]|十[一二三四五六七八九]?|二十[一二三四]?))[点时:：](?P<guo>超过|差|少|过|多)(?P<minute>[0-9一二三四五六七八九十零]+)分.*?")
    hour_guo_ke_pat = re.compile(u".*?([01]?\\d|2[0-3]|([一二三四五六七八九两]|十[一二三四五六七八九]?|二十[一二三四]?))[点时:：](?P<guo>超过|差|少|过|多)(?P<minute>[0-9一二三四五六七八九十零]+)刻.*?")
    hour_minute_pat = re.compile(u".*?([\\d零一二三四五六七八九十两]+)[点时:：](?P<minute>[0-9一二三四五六七八九十零]+)分.*?")
    hour_ke_pat = re.compile(u".*?([0-9一二三四五六七八九十零]+)[点时:：](?P<minute>[一二三123])刻.*?")
    hour_pat = re.compile(u".*?([01]?\\d|2[0-4]|[一二三四五六七八九两零]|十[一二三四五六七八九]?|二十[一二三四]?)[点时:：](半|零[0-9]|[0-5]?\\d|[零一二三四五]十?[一二三四五六七八九]|[一二三四五]十|十[一二三四五六七八九]).*?")
    qunima_pat = re.compile(u".*?(?P<guo>差)(?P<minute>[0-9一二三四五六七八九十零]+)分钟?到?(?P<dian>[01]?\\d|2[0-3]|([一二三四五六七八九两]|十[一二三四五六七八九]?|二十[一二三四]?))[点时].*?")
    hour_xu_pat = re.compile(u".*?([01]?\\d|2[0-4]|[零一二三四五六七八九两]|十[一二三四五六七八九]?|二十[一二三四]?)[点时][整许]?.*?")
    gundan_pat = re.compile(u".*?(?P<guo>差)(?P<minute>[0-9一二三四五六七八九十零]+)刻钟?到?(?P<dian>[01]?\\d|2[0-3]|([一二三四五六七八九两]|十[一二三四五六七八九]?|二十[一二三四]?))[点时].*?")
    relative_hour_pat = re.compile(u'.*?(上|下){0,}(本|上+|下+|这)(一?个?)小时.*?')
    relative_hour_diviation_pat = re.compile(u'.*?(?<![\\d])([0-9])个?小时(之前|之后|前|后).*?')

    def get_relative_hour(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        matcher = self.relative_hour_pat.match(entity)
        decorate_deviation,deviation = '',''
        time = arrow.get(basetime)
        base_hour = time.hour
        util = util_tools.Util()
        if matcher:
            if matcher.group(1):
                decorate_deviation = matcher.group(1)
            util = util_tools.Util()
            realHour = base_hour + util.rel_deviation(matcher.group(2), decorate_deviation)
            if commonParser:
                commonParser.timeUnit[5] = True
            return realHour
        else:
            matcher = self.relative_hour_diviation_pat.match(entity)
            if matcher:
                hour = base_hour + util.rel_deviation_num(matcher.group(2), int(matcher.group(1)))
                if commonParser:
                    commonParser.timeUnit[5] = True
                return hour
        return None

    def get_absolute_hour(self, entity, commonParser=None):
        hour_xu_matcher = self.hour_xu_pat.match(entity)
        qunima_matcher = self.qunima_pat.match(entity)
        gundan_matcher = self.gundan_pat.match(entity)
        hour_minute_guo_matcher = self.hour_minute_guo_pat.match(entity)
        hour_guo_ke_matcher = self.hour_guo_ke_pat.match(entity)
        hour_minute_matcher = self.hour_minute_pat.match(entity)
        hour_ke_matcher = self.hour_ke_pat.match(entity)
        hour_matcher = self.hour_pat.match(entity)
        apm = Apm()
        if qunima_matcher:
            minuteToParse = digitconv.getNumFromHan(qunima_matcher.group("minute"))
            chhour = qunima_matcher.group(4)
            hour_ = digitconv.getNumFromHan(chhour)
            minute_ = digitconv.getNumFromHan(minuteToParse)
            if qunima_matcher.group("guo"):
                guocha = qunima_matcher.group("guo")
                if guocha == u'差':
                    hourcha = minute_ / 60;
                    hour_ = hour_ - hourcha - 1;
                    minute_ = 60 - minute_ % 60;
                    return None
            if commonParser:
                if commonParser.timeUnit[4]:
                    hour_ = apm.adjustHours(entity, hour_, commonParser)
                commonParser.timeUnit[5] = True
                commonParser.timeUnit[6] = True
                if minute_ == 60:
                    hour_ = hour_ + 1
                elif 0 <= minute_ < 60:
                    commonParser.date = commonParser.date.replace(hour=hour_).replace(minute=minute_)
                    return None
                else:
                    commonParser.timeFormatInfo = u'时间表述异常，分钟应该在[0,60]范围内！'
                    return None

                if hour_ == 24:
                    commonParser.date = commonParser.date.replace(days=commonParser.date.day + 1).replace(
                        hour=0).replace(minute=0)
                else:
                    if 0 <= hour_ < 24:
                        commonParser.date = commonParser.date.replace(
                            hour=hour_).replace(minute=0)
                    else:
                        commonParser.timeFormatInfo = u'时间表述异常，小时应该在[0,24]范围内！'
                        return None
            return hour_

        elif gundan_matcher:
            minuteToParse = digitconv.getNumFromHan(qunima_matcher.group("minute"))
            if minuteToParse != 0:
                minute_ = minuteToParse * 15
            chhour = qunima_matcher.group(4)
            hour_ = digitconv.getNumFromHan(chhour)
            if gundan_matcher.group("guo"):
                guocha = qunima_matcher.group("guo")
                if guocha == u'差':
                    hourcha = minute_ / 60
                    hour_ = hour_ - hourcha - 1
                    minute_ = 60 - minute_ % 60
                    return None
            if commonParser:
                if commonParser.timeUnit[4]:
                    hour_ = apm.adjustHours(entity, hour_, commonParser)
                commonParser.timeUnit[5] = True
                commonParser.timeUnit[6] = True
                if minute_ == 60:
                    hour_ = hour_ + 1
                elif 0 <= minute_ < 60:
                    commonParser.date = commonParser.date.replace(hour=hour_).replace(minute=minute_)
                    return None
                else:
                    commonParser.timeFormatInfo = u'时间表述异常，分钟应该在[0,60]范围内！'
                    return None

                if hour_ == 24:
                    commonParser.date = commonParser.date.replace(days=commonParser.date.day + 1).replace(
                        hour=0).replace(minute=0)
                else:
                    if 0 <= hour_ < 24:
                        commonParser.date = commonParser.date.replace(
                            hour=hour_).replace(minute=0)
                    else:
                        commonParser.timeFormatInfo = u'时间表述异常，小时应该在[0,24]范围内！'
                        return None
            return hour_

        elif hour_minute_guo_matcher:
            hourToParse = hour_minute_guo_matcher.group(1)
            hour_ = digitconv.getNumFromHan(hourToParse)
            minute_ = digitconv.getNumFromHan(hour_minute_guo_matcher.group("minute"))
            if hour_minute_guo_matcher.group("guo"):
                guocha = hour_minute_guo_matcher.group("guo")
                if guocha == u"差" or  guocha == u"少":
                    hourcha = minute_ / 60
                    hour_ = hour_ - hourcha-1
                    minute_ = 60-minute_ % 60
            if commonParser:
                if commonParser.timeUnit[4]:
                    hour_ = apm.adjustHours(entity, hour_, commonParser)
                commonParser.timeUnit[5] = True
                commonParser.timeUnit[6] = True
                if minute_ == 60:
                    hour_ = hour_ + 1
                elif 0 <= minute_ < 60:
                    commonParser.date = commonParser.date.replace(hour=hour_).replace(minute=minute_)
                    return None
                else:
                    commonParser.timeFormatInfo = u'时间表述异常，分钟应该在[0,60]范围内！'
                    return None

                if hour_ == 24:
                    commonParser.date = commonParser.date.replace(days=commonParser.date.day + 1).replace(
                        hour=0).replace(minute=0)
                else:
                    if 0 <= hour_ < 24:
                        commonParser.date = commonParser.date.replace(
                            hour=hour_).replace(minute=0)
                    else:
                        commonParser.timeFormatInfo = u'时间表述异常，小时应该在[0,24]范围内！'
                        return None
            return hour_

        elif hour_guo_ke_matcher:
            chhour = hour_guo_ke_matcher.group(1)
            hour_ = digitconv.getNumFromHan(chhour)
            minuteToParse = digitconv.getNumFromHan(hour_guo_ke_matcher.group("minute"))
            minute_ = minuteToParse * 15
            if hour_guo_ke_matcher.group("guo"):
                guocha = hour_guo_ke_matcher.group("guo")
                if guocha == u"差" or guocha == u"少":
                    hourcha = minute_ / 60
                    hour_ = hour_ - hourcha - 1
                    minute_ = 60 - minute_ % 60
            if commonParser:
                if commonParser.timeUnit[4]:
                    hour_ = apm.adjustHours(entity, hour_, commonParser)
                commonParser.timeUnit[5] = True
                commonParser.timeUnit[6] = True
                if minute_ == 60:
                    hour_ = hour_ + 1
                elif 0 <= minute_ < 60:
                    commonParser.date = commonParser.date.replace(hour=hour_).replace(minute=minute_)
                    return None
                else:
                    commonParser.timeFormatInfo = u'时间表述异常，分钟应该在[0,60]范围内！'
                    return None

                if hour_ == 24:
                    commonParser.date = commonParser.date.replace(days=commonParser.date.day + 1).replace(
                        hour=0).replace(minute=0)
                else:
                    if 0 <= hour_ < 24:
                        commonParser.date = commonParser.date.replace(
                            hour=hour_).replace(minute=0)
                    else:
                        commonParser.timeFormatInfo = u'时间表述异常，小时应该在[0,24]范围内！'
                        return None
            return hour_

        elif hour_minute_matcher:
            chhour = hour_minute_matcher.group(1)
            hour_ = digitconv.getNumFromHan(chhour)
            chminute = hour_minute_matcher.group("minute")
            minute_ = digitconv.getNumFromHan(chminute)
            if commonParser:
                if commonParser.timeUnit[4]:
                    hour_ = apm.adjustHours(entity, hour_, commonParser)
                commonParser.timeUnit[5] = True
                commonParser.timeUnit[6] = True
                if minute_ == 60:
                    hour_ = hour_+1
                elif 0 <= minute_ < 60:
                    commonParser.date = commonParser.date.replace(hour=hour_).replace(minute=minute_)
                    return None
                else:
                    commonParser.timeFormatInfo = u'时间表述异常，分钟应该在[0,60]范围内！'
                    return None

                if hour_ == 24:
                    commonParser.date = commonParser.date.replace(days=commonParser.date.day+1).replace(hour=0).replace(minute=0)
                else:
                    if 0 <= hour_ < 24:
                        commonParser.date = commonParser.date.replace(
                            hour=hour_).replace(minute=0)
                    else:
                        commonParser.timeFormatInfo = u'时间表述异常，小时应该在[0,24]范围内！'
                        return None
            return hour_

        elif hour_ke_matcher:
            hour_ = digitconv.getNumFromHan(hour_ke_matcher.group(1))
            chminute = hour_ke_matcher.group("minute")
            minute_ = digitconv.getNumFromHan(chminute) * 15
            if commonParser:
                if commonParser.timeUnit[4]:
                    hour_ = apm.adjustHours(entity, hour_, commonParser)
                commonParser.timeUnit[5] = True
                commonParser.timeUnit[6] = True
                if minute_ == 60:
                    hour_ = hour_+1
                elif 0 <= minute_ < 60:
                    commonParser.date = commonParser.date.replace(hour=hour_).replace(minute=minute_)
                    return None
                else:
                    commonParser.timeFormatInfo = u'时间表述异常，分钟应该在[0,60]范围内！'
                    return None

                if hour_ == 24:
                    commonParser.date = commonParser.date.replace(days=commonParser.date.day+1).replace(hour=0).replace(minute=0)
                else:
                    if 0 <= hour_ < 24:
                        commonParser.date = commonParser.date.replace(
                            hour=hour_).replace(minute=0)
                    else:
                        commonParser.timeFormatInfo = u'时间表述异常，小时应该在[0,24]范围内！'
                        return None
            return hour_

        elif hour_matcher:
            chhour = hour_matcher.group(1)
            hour_ = digitconv.getNumFromHan(chhour)
            minuteToParse = hour_matcher.group(2)
            hour_ = apm.adjustHours(entity, hour_, commonParser)
            if minuteToParse == u"半":
                minute_ = 30
            else:
                minute_ = digitconv.getNumFromHan(minuteToParse)
            if commonParser:
                commonParser.timeUnit[5] = True
                commonParser.timeUnit[6] = True
                commonParser.date = commonParser.date.replace(hour=hour_).replace(minute=minute_)
            return hour_

        elif hour_xu_matcher:
            chhour = hour_xu_matcher.group(1)
            hour_ = digitconv.getNumFromHan(chhour)
            if commonParser:
                if commonParser.timeUnit[4]:
                    hour_ = apm.adjustHours(entity, hour_, commonParser)
                commonParser.timeUnit[5] = True
                if u'许' in entity or u'整' in entity:
                    commonParser.timeUnit[6] = True
                    commonParser.timeUnit[7] = True
                if hour_ == 24:
                    commonParser.date = commonParser.date.shift(days=1).replace(hour=0)
                else:
                    commonParser.date = commonParser.date.replace(hour=hour_)
            return hour_
        return None

    def get_hour(self, entity, basetime='2018-11-01 12:00:00', commonParser=None):
        hour = self.get_relative_hour(entity, basetime, commonParser)
        if not hour is None:
            return hour
        return self.get_absolute_hour(entity, commonParser)

if __name__ == '__main__':
    hour_proc = Hour()
    assert hour_proc.get_hour(u'2007年3月21号') is None

    assert hour_proc.get_hour(u'3点过5分') == 3

    file_list = [u'../test/hour_testcase.csv']

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
                                    day = hour_proc.get_hour(entity, basetime)
                                elif abs_rel == u'absolute':
                                    day = hour_proc.get_hour(entity)

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
