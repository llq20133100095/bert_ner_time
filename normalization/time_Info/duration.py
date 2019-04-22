# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 21:01:32 2019

@author: leolqli
@Function: process the "time_duration" type. It is a basic class.
"""
import re
from utils import digitconv

class Duration:
    def __init__(self):
        self.pattern_num_point = u'.*?([0-9零一二三四五六七八九十两]+)\.([0-9零一二三四五六七八九十两]+).*?' \
            + u'(世纪|年|周年|月|星期|礼拜|周末|周|天|工作日|日|夜|晚|早|小时|分钟|刻钟|秒).*?'
        self.pattern_num_half = u'.*?([0-9零一二三四五六七八九十两]+)(个半|整).*?' \
            + u'(世纪|年|周年|月|星期|礼拜|周末|周|天|工作日|日|夜|晚|早|小时|分钟|刻钟|秒).*?'
        self.pattern_num = u'.*?([0-9零一二三四五六七八九十两半整]+).*?' \
            + u'(世纪|年|周年|月|星期|礼拜|周末|周|天|工作日|日|夜|晚|早|小时|分钟|刻钟|秒).*?'
        
        #store the time, (list_index, basics_value)
        self.duration_dict = {u'世纪':(0, 100), u'年':(0, 1), u'周年':(0, 1), \
              u'月':(1, 1), u'星期':(2, 1), u'礼拜':(2, 1), u'周':(2, 1), u'天':(3, 1), \
              u'工作日':(3, 1), u'周末':(3, 2), u'夜':(3, 1), u'晚':(3, 1), u'小时':(4, 1), \
              u'分钟':(5, 1), u'秒':(6, 1), u'日':(3, 1), u'刻钟':(5, 15), u'早':(3, 1)}
        
    def duration_output(self, number, suffix_word):
        """
        Function:
            Process the time duration, and get the output of the duration.
        Parameters:
            1.number: str, samples like the "2"; list, [number1, number2] and samples like "2.3";
                list, [number1, '半'] and samples like the "一个半"
            2.suffix_word: str, "世纪|年|周年|月|星期|礼拜|周末|周|天|工作日|夜|晚|小时|分钟|秒"
        Return:
            1.time_dur_list: list, represent the "年、月、周、日、时、分、秒" 
        """
        time_dur_list = [0, 0, 0, 0, 0, 0, 0]
        shift = 0.0
        if(type(number) == list):
            if(number[1] == u'个半'):
                #process the "一个半"
                shift = digitconv.getNumFromHan(number[0]) + 0.5
            elif(number[1] == u'整'):
                shift = digitconv.getNumFromHan(number[0])
            else:
                shift = str(digitconv.getNumFromHan(number[0])) + "." + \
                    str(digitconv.getNumFromHan(number[1]))
                shift = float(shift)

        else:
            if(number == u'半'):
                shift = 0.5
            elif(number == u'整'):
                shift = 1.0
            else:
                shift = float(digitconv.getNumFromHan(number))
        
        time_indx = self.duration_dict[suffix_word]
        time_dur_list[time_indx[0]] = shift * time_indx[1]
            
        return time_dur_list
    
    def duration_recognise(self, entity):
        """
        Function:
            recognise the "time_duration" type in which pattern
        Parameters:
            1.entity: str,
        Return:
            1.time_dur_list: list, represent the "年、月、周、日、时、分、秒"
            2.error: boolean, "True" value have no this patter
        """
        time_dur_list = [0, 0, 0, 0, 0, 0, 0]
        error = False

        matcher_num_point = re.match(self.pattern_num_point, entity) 
        matcher_num_half = re.match(self.pattern_num_half, entity)
        matcher_num = re.match(self.pattern_num, entity)  

        
        if(matcher_num_point):
            number = [matcher_num_point.group(1), matcher_num_point.group(2)]
            time_dur_list = self.duration_output(number, matcher_num_point.group(3))

        elif(matcher_num_half):
            number = [matcher_num_half.group(1), matcher_num_half.group(2)]
            time_dur_list = self.duration_output(number, matcher_num_half.group(3))            
        
        elif(matcher_num):
            time_dur_list = self.duration_output(matcher_num.group(1), matcher_num.group(2))            
            
        else:
            print("-----------------------------")
            print("no matcher in time_duration")
            error = True
            
        return time_dur_list, error
    
if __name__ == "__main__":
    duration = Duration()
    entity = u'一个早上'
    time_fre_list, error = duration.duration_recognise(entity)
