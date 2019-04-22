# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 22:21
# @Author  : honeyding
# @FileName: text_process.py

import unittest
import digitconv

class Util():
    lunar_date = {u'正月':1, u'腊月':11, u'冬月':12, u'大年':1, u'元月':1}

    def rel_deviation(self, deviation, decorate_deviation):
        length = len(decorate_deviation)
        time_val = 0
        if deviation == u'前':
            time_val = -2 + -1 * length
        elif deviation in [u'去', u'昨', u'上', u'上个']:
            time_val = -1 + -1 * length
        elif deviation in [u'今', u'本', u'当', u'这个', u'同', u'这', u'那', u'那个']:
            time_val = 0
        elif deviation in [u'明', u'次', u'下', u'下个', u'下一', u'翌', u'来']:
            time_val = 1 + length
        elif deviation == u'后':
            time_val = 2 + length
        return time_val

    def rel_deviation_num(self, deviation, drift_num):
        val = 0
        if deviation in [u'前', u'之前']:
            val = (-1) * drift_num
        elif deviation in [u'之后', u'后']:
            val = +drift_num
        return val

    def week_deviation_num(self, deviation, week_day):
        if week_day == u'天' or week_day == u'日':
            week_day_num = 7
        else:
            week_day_num = digitconv.getNumFromHan(week_day)
        val = 0
        if deviation:
            if u'上' in deviation:
                val = (-1) * 7 * len(deviation)
            elif u'下' in deviation:
                val = 7 * len(deviation)
        return val + week_day_num - 1

    def lunar_month_num(self, lunar_month):
        if self.lunar_date.has_key(lunar_month):
            return self.lunar_date.get(lunar_month)


if __name__ == '__main__':
    util = Util()
    assert util.getDeviation(u'后', u'大') == 3
    assert util.getDeviation(u'前', u'大') == -3
    assert util.getDeviation(u'今', '') == 0
    assert util.getDeviation(u'上', u'上上') == -3
