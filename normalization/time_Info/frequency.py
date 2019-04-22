# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:34:37 2019

@author: leolqli
@Function: process the time frequency type. It is a basic class.
"""
import re
from utils import digitconv

class Frequency:
    def __init__(self):
        
        self.pattern_whenever = u'.*?(每逢).*?(年|寒暑假|季度|月|周|星期|礼拜|日|天|晚|中午|早).*?'
        self.pattern_each_num = u'.*?(每)(隔*)([0-9零一二两三四五六七八九|半]{1,}).*?(年|寒暑假|季度|月|周|星期|礼拜|日|天|晚|时|分|秒|中午|早).*?'
        self.pattern_each = u'.*?(每)(隔*).*?(年|寒暑假|季度|月|周|星期|礼拜|日|天|晚|时|分|秒|中午|早).*?'
        self.pattern_dayday = u".*?(天天).*?([晚|中午|早]{0,}).*?"
        self.pattern_spe_word = u'.*?(年年|岁岁年年|日日夜夜|月月).*?'
        
        #store the time, (list_index, basics_value)
        self.fre_indx_dict = {u'世纪':(0, 100), u'年':(0, 1), u'寒暑假':(0, 0.5), u'季度':(1, 3), \
              u'月':(1, 1), u'周':(2, 1), u'星期':(2, 1), u'礼拜':(2, 1), \
              u'日':(3, 1), u'天':(3, 1), u'晚':(3, 1), u'时':(4, 1), u'分':(5, 1), \
              u'秒':(6, 1), u'中午':(3, 1), u'早':(3, 1), u'年年':(0, 1), \
              u'岁岁年年':(0, 1), u'日日夜夜':(3, 1), u'月月':(1, 1), u'天天':(3, 1)}
        
        self.fre_indx_whenever = {u'年':(0, 1), u'寒暑假':(0, 1), u'季度':(0, 1), \
              u'月':(0, 1), u'周':(2, 1), u'星期':(2, 1), u'礼拜':(2, 1), \
              u'日':(3, 1), u'天':(3, 1), u'晚':(3, 1), u'中午':(3, 1), u'早':(3, 1)}
    
    def frequency_output(self, prefix_word, number, suffix_word):
        """
        Function:
            Process the time frequency, and get the output of the frequency.
        Parameters:
            1.prefix_word: str, "每|每隔"
            2.number: str, "个|0-9零一二三四五六七八九|半"
            3.suffix_word: str, "年|寒暑假|季度|月|周|星期|礼拜|日|晚上|时|分|秒"
        Return:
            1.time_fre_list: list, represent the "年、月、周、日、时、分、秒" 
        """
        time_fre_list = [0, 0, 0, 0, 0, 0, 0]
        
        if(prefix_word == u'每逢'):
            time_indx = self.fre_indx_whenever[suffix_word]
            time_fre_list[time_indx[0]] = time_indx[1]
            
        else:
            shift = 0.0
            if(prefix_word == u'每隔'):
                shift = 1.0
            
            fre_num = 1.0
            try:
                fre_num = int(digitconv.getNumFromHan(number))
            except:
                if(number == u'半' and prefix_word == u'每隔'):
                    fre_num = 0.5
                    shift = 0.5
                elif(number == u'半'):
                    fre_num = 0.5            
            
            time_indx = self.fre_indx_dict[suffix_word]
            time_fre_list[time_indx[0]] = shift + fre_num * time_indx[1]
        return time_fre_list
            
    def frequency_recognise(self, entity):
        """
        Function:
            recognise the "time_frequency" type in which pattern
        Parameters:
            1.entity: str,
        Return:
            1.time_fre_list: list, represent the "年、月、周、日、时、分、秒"
            2.error: boolean, "True" value have no this patter
        """
        time_fre_list = [0, 0, 0, 0, 0, 0, 0]
        error = False

        matcher_whenever = re.match(self.pattern_whenever, entity)  
        matcher_each_num = re.match(self.pattern_each_num, entity)        
        matcher_each = re.match(self.pattern_each, entity)
        matcher_dayday = re.match(self.pattern_dayday, entity)
        matcher_spe_word = re.match(self.pattern_spe_word, entity)  

        if(matcher_whenever):
             time_fre_list = self.frequency_output(matcher_whenever.group(1), '', matcher_whenever.group(2))                       
        
        elif(matcher_each_num):
            prefix_word = matcher_each_num.group(1) + matcher_each_num.group(2)
            time_fre_list = self.frequency_output(prefix_word, matcher_each_num.group(3), matcher_each_num.group(4))
            
        elif(matcher_each):
            prefix_word = matcher_each.group(1) + matcher_each.group(2)
            time_fre_list = self.frequency_output(prefix_word, '', matcher_each.group(3))
        
        elif(matcher_dayday):
            time_fre_list = self.frequency_output('', '', matcher_dayday.group(1))
        
        elif(matcher_spe_word):
            time_fre_list = self.frequency_output('', '', matcher_spe_word.group(1))            
        
        else:
            print("-----------------------------")
            print("no matcher in time_frequent")
            error = True
            
        return time_fre_list, error
        
if __name__ =='__main__':
    frequency = Frequency()
    entity = u'岁岁年年'
    time_fre_list, error = frequency.frequency_recognise(entity)