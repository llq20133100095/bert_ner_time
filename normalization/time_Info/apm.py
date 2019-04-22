#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/24 16:33
# @Author  : honeyding
# @File    : apm.py
# @Software: PyCharm

import re

class Apm:
    apm_pat = re.compile(u'.*?(明早|傍晚|早上|早晨|凌晨|上午|中午|下午|大晚上|晚上|夜里|今晚|明晚|昨晚|前晚|这晚|晚|清晨|午后).*?')
    apm_hour_pat = re.compile(u'.*?(明早|傍晚|早上|早晨|凌晨|上午|中午|下午|大晚上|晚上|夜里|今晚|明晚|昨晚|前晚|这晚|晚|清晨|午后).*?([0-9一二三四五六七八九两十]).*?')


    def get_apm_info(self, entity, commonParser):
        matcher = self.apm_pat.match(entity)
        if matcher:
            if commonParser:
                commonParser.timeUnit[4] = True
            return True
        return False
    
    def judge_apm_hour(self, entity, commonParser):
        matcher = self.apm_hour_pat.match(entity)
        if matcher:
            if commonParser:
                commonParser.timeUnit[4] = True
            return True
        return False

    def adjustHours(self, entity, hour, commonParser):
        if u"早" not in entity and u"上午" not in entity and u"晨" not in entity:
            if u"中午" in entity:
                if hour > 14 or hour > 2 and hour < 10:
                    print(u'不能是中午。')
                    commonParser.timeAPMInfo = str(hour) + u"点不能是中午。"
                elif hour < 2 and hour > 0:
                    hour += 12

            elif u"下午" not in entity and u"午后" not in entity:
                if u"昨晚" in entity or u"明晚" in entity or u"傍晚" in entity or u"晚" in entity or u"晚上" in entity or u"夜里" in entity or u"今晚" in entity:
                    if hour > 12 and hour < 17  or  hour >= 0 and hour < 5:
                        print(u'不能是晚上。')
                        commonParser.timeAPMInfo = str(hour) + u"点不能是晚上。"
                    elif hour >= 4 and hour <= 12:
                        hour += 12

            else:
                if hour > 0 and hour <= 12:
                    hour += 12
                # if hour > 19 or hour < 1 or hour > 7 and hour < 12:
                #     print(u'不能是下午。')
                #     commonParser.timeAPMInfo = str(hour) + u'不能是下午。'
                # elif hour > 0 and hour <= 7:
                #     hour += 12

        elif hour > 12:
            print(u'不能是上午或早上。')
            commonParser.timeAPMInfo = str(hour) + u'点不能是上午或早上。'
        return hour

if __name__ == '__main__':
    apm_proc = Apm()
    assert apm_proc.get_apm_info(u'早晨') is True