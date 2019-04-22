# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:38:39 2019

@author: leolqli
@function:
    1.process has the “前”、“后”、“未来”、“明”、“后期”、“前后”、“里”、“开始”、“上”、“下” word
    2.time:世纪、年、月、日|天、apm、点|时、分|分钟|刻、秒
"""
import re
from utils import digitconv
import arrow
import json

from CommonParser import CommonParser
from common.entity import entity as Entity
from time_slot_value import TimeSlotValue

class TimePeriodAround:
    
    def __init__(self):
        self.pattern_year = u'.*?([0-9]{4})年.*?'
        self.pattern_year2 = u'.*?([0-9]{4})-([0-9]{2}).*?'
        self.pattern_half_double_year = u'.*?(年).*?(年).*?(来|以来)'
        
        self.pattern_half = u'.*?(前|后|上|下)(半).*?(世纪|年)'
        self.pattern_half_since = u'.*?(半).*?(世纪|年|月)(来|以来|里|内)'
        self.pattern_around_num_prefix = u'.*?(?<![晚上|节前|早上])(前|后|未来|明|上|下|昨|今|去|这|过去)([0-9零一二三四五六七八九十两]{1,}).*?(世纪|年|月|日|天|晚|点|时|分|分钟|刻|秒)([中|来|以来|里]{0,})$'
        self.pattern_around_num_prefix_age_reach = u'.*?(上|下|这).*?(世纪)([0-9零一二三四五六七八九十两]{1,}).*?(年代).*?(到|至|和|-|~|—|～|及).*?(上|下|这).*?(世纪)([0-9零一二三四五六七八九十两]{1,}).*?(年代)'
        self.pattern_around_num_prefix_age_century_reach = u'.*?(上|下|这).*?(世纪)([0-9零一二三四五六七八九十两]{1,}).*?(年代).*?(到|至|和|-|~|—|～|及).*?(本|今).*?(世纪|年|月|日|天|晚).*?(里|内|中|前期|初期|中期|后期|末).*?'
        self.pattern_around_num_prefix_age_suffix_reach = u'.*?(上|下|这).*?(世纪)([0-9零一二三四五六七八九十两]{1,}).*?(年代).*?(到|至|和|-|~|—|～|及).*?([0-9零一二三四五六七八九十两]{1,}).*?(世纪|年|月|日|天|晚|点|时|分|分钟|刻|秒).*?(?!中旬)(里|内|中|前期|初期|中期|后期|末).*?'

        self.pattern_around_num_prefix_age_suffix = u'.*?(上|下|这).*?(世纪)([0-9零一二三四五六七八九十两]{1,}).*?(年代)(前期|初期|中期|后期|末)'
        self.pattern_around_num_prefix_age2 = u'.*?(上|下|这).*?(世纪)([0-9零一二三四五六七八九十两]{1,})、([0-9零一二三四五六七八九十两]{1,}).*?(年代)'
        self.pattern_around_num_prefix_age = u'.*?(上|下|这).*?(世纪)([0-9零一二三四五六七八九十两]{1,}).*?(年代)'
        self.pattern_around_num_suffix = u'.*?([0-9零一二三四五六七八九十两]{1,}).*?(世纪|年|月|日|天|晚|点|时|分|分钟|刻|秒).*?(?![月末|月中|中旬|中午])(里|内|中|前期|初期|中期|后期|末|来|间).*?'
        self.pattern_around_suffix = u'(?!周).*?(本|今|全).*?(世纪|年|月|日|天|晚).*?(?!中旬|中午)(里|内|中|前期|初期|中期|后期|末|度{0,})(.*)'
        self.pattern_around_only_time = u'.*?([0-9零一二三四五六七八九十两]{1,}).*?(世纪)'
        self.pattern_this = u''

        #load the time around word
        file_around = open("./resources/time_around_word.json", "r")
        self.around_dict = json.load(file_around)
        
#        #get the apm time
#        file_apm = open("./resources/time_duration_apm.json", "r")
#        self.apm_dict = json.load(file_apm)
#        apm_list = list(self.apm_dict.keys())
#        apm_list.sort(key = lambda i: len(i), reverse=True)
#        self.pattern_apm = u'(' + '|'.join(apm_list) + u')'
        
        #save the 前期|中期|后期 in "世纪"
        self.century_period = {u'前期':(0, 29), u'初期':(0, 29), u'中期':(30, 69), \
           u'后期':(70, 99), u'末':(70, 99)}
        
        #save the 前期|中期|后期 in "年代"
        self.year_stage = {u'年代前期':(0, 3), u'年代初期':(0, 3), u'年代中期':(3, 6), \
           u'年代后期':(7, 9), u'年代末':(7, 9)}
        
        
    def switch_time(self, shift_time, start_date, end_date):
        """
        Function:
            1.when the shift_time > 0, switch the value between the start_date and end_date
        Parameter:
            1.shift_time: int, >0 or <0
            2.start_date: arrow
            3.end_date: arrow
        Return:
            start_date, end_date
        """
        if(shift_time > 0):
            temp_date = start_date
            start_date = end_date
            end_date = temp_date
        
        return start_date, end_date

    def set_timeUnit(self, com_par, time_word):
        """
        Function:
            set the timeUnit according to the time_word
        Paramters:
            1.com_par: CommonParser
            2.time_word: str, 世纪|年|月|日|天|晚|点|时|分|分钟|刻|秒
        Return:
            com_par
        """
        if time_word in u"世纪" or u"年代" in time_word:
            com_par.timeUnit[0] = True
        if time_word in u"年":
            com_par.timeUnit[1] = True
        if time_word in u"月":
            com_par.timeUnit[2] = True
        if time_word in u"日天":
            com_par.timeUnit[3] = True
        if time_word in u"晚":
            com_par.timeUnit[4] = True
        if time_word in u"点时":
            com_par.timeUnit[5] = True
        if time_word in u"分钟刻":
            com_par.timeUnit[6] = True
        if time_word in u"秒":
            com_par.timeUnit[7] = True   
    
        return com_par
    
    def cal_integer_time_prefix_parse(self, time_around_prefix, time_number, time_word, basetime, com_par):
        """
        Function:
            1.calculate the number with time_word
            2.process the "年代" word and "年代前期" word
        Parameter:
            1.time_around_prefix: str, "前|后|未来|明|上|下|昨|今|去|这"
            2.time_number: str, 0-9
            3.time_word: str, 世纪|年|月|日|天|晚|点|时|分|分钟|刻|秒, 年代, 年代前期
            4.basetime: str,
            5.com_par: CommonParser
        """
        #solve the "七八十"
        if(u'年代' in time_word and len(time_number)==3):
            start_time_number = digitconv.getNumFromHan(time_number[0] + time_number[2])
            end_time_number = digitconv.getNumFromHan(time_number[1] + time_number[2])
        else:
            time_number = digitconv.getNumFromHan(time_number)
        
        #process the "世纪、年、月、日、apm、时、分、秒"
        com_par.timeUnit = [False,False,False,False,False,False,False,False]
        
        #set the timeUnit
        com_par = self.set_timeUnit(com_par, time_word)
            
        start_date = arrow.get(basetime)
        end_date = arrow.get(basetime)
        
        #get the base number and shift_time
        time_around_base = self.around_dict[time_around_prefix][u'offset']
        shift_time = time_number * time_around_base
        
        #process the start and end date        
        if com_par.timeUnit[7]:
            start_date = start_date.shift(seconds=shift_time)
#            end_date = end_date.shift(seconds=time_around_base)
        if com_par.timeUnit[6]:
            start_date = start_date.shift(minutes=shift_time)
#            end_date = end_date.shift(minutes=time_around_base)
        if com_par.timeUnit[5]:
            start_date = start_date.shift(hours=shift_time)
#            end_date = end_date.shift(hours=time_around_base)

        start_date, end_date = self.switch_time(shift_time, start_date, end_date)
            
        #process "世纪、年、月、日"
        if com_par.timeUnit[4]:
            start_date = start_date.shift(days=shift_time)
            end_date = end_date.shift(days=time_around_base)
            
            #if have the "下、后", switch two value
            start_date, end_date = self.switch_time(shift_time, start_date, end_date)

            start_date = start_date.replace(hour=18).replace(minute=0).replace(second=0)
            end_date = end_date.replace(hour=23).replace(minute=59).replace(second=0)
            
        if com_par.timeUnit[3]:
            start_date = start_date.shift(days=shift_time)
            end_date = end_date.shift(days=time_around_base)
            
            #if have the "下、后", switch two value
            start_date, end_date = self.switch_time(shift_time, start_date, end_date)

            start_date = start_date.replace(hour=0).replace(minute=0).replace(second=0)
            end_date = end_date.replace(hour=23).replace(minute=59).replace(second=0)

        if com_par.timeUnit[2]:
            start_date = start_date.shift(months=shift_time)
            end_date = end_date.shift(months=time_around_base)

            #if have the "下、后", switch two value
            start_date, end_date = self.switch_time(shift_time, start_date, end_date)

            start_date = start_date.replace(day=1).replace(hour=0).replace(minute=0).replace(second=0)
            end_date = end_date.replace(day=end_date.ceil('month').day).replace(hour=23).replace(minute=59).replace(second=0)

        if com_par.timeUnit[1]:
            start_date = start_date.shift(years=shift_time)
            end_date = end_date.shift(years=time_around_base)

            #if have the "下、后", switch two value
            start_date, end_date = self.switch_time(shift_time, start_date, end_date)

            start_date = start_date.replace(month=1).replace(day=1).replace(hour=0).replace(minute=0).replace(second=0)
            end_date = end_date.replace(month=12).replace(day=31).replace(hour=23).replace(minute=59).replace(second=0)

        if com_par.timeUnit[0]:
            #get the base century
            century = start_date.year / 100

            if(u'年代' in time_word):
                century += time_around_base
                if(type(time_number) == type(u'')):
                    start_year = century * 100 + start_time_number
                    end_year = century * 100 + end_time_number + 9
                else:
                    start_year = century * 100 + time_number
                    end_year = start_year + 9
                    
                    #have "年代后期"
                    if(time_word in self.year_stage.keys()):
                        end_year = start_year + self.year_stage[time_word][1]
                        start_year += self.year_stage[time_word][0]
                    
                start_date = start_date.replace(year=start_year).replace(month=1).replace(day=1).replace(hour=0).replace(minute=0).replace(second=0)
                end_date = end_date.replace(year=end_year).replace(month=12).replace(day=31).replace(hour=23).replace(minute=59).replace(second=0)                
            else:
                century = century + shift_time
                start_date = start_date.replace(year=century * 100).replace(month=1).replace(day=1).replace(hour=0).replace(minute=0).replace(second=0)
                end_date = end_date.replace(year=century * 100 + 99).replace(month=12).replace(day=31).replace(hour=23).replace(minute=59).replace(second=0)
            
    
        com_par.startDate = start_date
        com_par.endDate = end_date
        return com_par
    
    def cal_half_time_parse(self, time_around_prefix, time_number, time_word, basetime, com_par):
        """
        Function:
            1.process the "半" case, such as "前半年"\"前半个世纪", "半年以来"
        Parameters:
            1.time_around_prefix: str, "前|后|上|下" or "以来"
            2.time_number: str, 0-9
            3.time_word: str, 世纪|年
            4.basetime: str,
            5.com_par: CommonParser
        """
        if(u'半' in time_number):
            time_number = 0.5
            
        #process the "世纪、年、月、日、apm、时、分、秒"
        com_par.timeUnit = [False,False,False,False,False,False,False,False]
        
        #set the timeUnit
        com_par = self.set_timeUnit(com_par, time_word)        

        start_date = arrow.get(basetime)
        end_date = arrow.get(basetime)
            
        if(time_around_prefix in u'以来'):
            if com_par.timeUnit[2]:
                start_date = start_date.shift(days=-15)
                
            if com_par.timeUnit[1]:
                start_date = start_date.shift(months=-6)
                
            if com_par.timeUnit[0]:
                start_date = start_date.shift(years=-50)
                
        elif(time_around_prefix in u'里' or time_around_prefix in u'内'):
            if com_par.timeUnit[2]:
                start_date = start_date.replace(hour=0, minute=0, second=0)
                end_date = start_date.shift(days=15)
                end_date = end_date.replace(hour=23, minute=59, second=0)
        else:
            #get the base number and shift_time
            time_around_base = self.around_dict[time_around_prefix][u'offset']
                
            if com_par.timeUnit[1]:
                #have the "前""上"
                if(time_around_base < 0):
                    start_date = start_date.replace(month=1).replace(day=1).replace(hour=0).replace(minute=0).replace(second=0)
                    end_date = end_date.replace(month=5).replace(day=31).replace(hour=23).replace(minute=59).replace(second=0)
                else:
                    start_date = start_date.replace(month=6).replace(day=1).replace(hour=0).replace(minute=0).replace(second=0)
                    end_date = end_date.replace(month=12).replace(day=31).replace(hour=23).replace(minute=59).replace(second=0)
                    
            if com_par.timeUnit[0]:
                #get the base century
                century = start_date.year / 100
                century = century * 100
                
                #have the "前""上"
                if(time_around_base < 0):
                    start_date = start_date.replace(year=century).replace(month=1).replace(day=1).replace(hour=0).replace(minute=0).replace(second=0)
                    end_date = end_date.replace(year=century + 49).replace(month=12).replace(day=end_date.ceil('month').day).replace(hour=23).replace(minute=59).replace(second=0)
                
                else:
                    start_date = start_date.replace(year=century + 50).replace(month=1).replace(day=1).replace(hour=0).replace(minute=0).replace(second=0)
                    end_date = end_date.replace(year=century + 99).replace(month=12).replace(day=end_date.ceil('month').day).replace(hour=23).replace(minute=59).replace(second=0)
                    
        com_par.startDate = start_date
        com_par.endDate = end_date
        return com_par

    def cal_integer_time_suffix_parse(self, time_around_suffix, time_number_this, time_word, basetime, com_par):
        """
        Function:
            1.calculate the number with time_word
        Parameter:
            1.time_around_suffix: str, "里|内|前期|中期|后期|度|来"
            2.time_number_this: str, 0-9 or "本|今|全"
            3.time_word: str, 世纪|年|月|日|天|晚|点|时|分|分钟|刻|秒
            4.basetime: str,
            5.com_par: CommonParser
        """
        if(time_number_this not in u'本今全'):
            time_number_this = digitconv.getNumFromHan(time_number_this)
        else:
            time_number_this = 1
        
        #process the "世纪、年、月、日、apm、时、分、秒"
        com_par.timeUnit = [False,False,False,False,False,False,False,False]
        
        #set the timeUnit
        com_par = self.set_timeUnit(com_par, time_word)
            
        start_date = arrow.get(basetime)
        end_date = arrow.get(basetime)
        
        #process the start and end date
        if(u'里' in time_around_suffix or u'内' in time_around_suffix or \
           u'中' in time_around_suffix or u'度' in time_around_suffix or \
           ""  == time_around_suffix or u'间' in time_around_suffix):
            if com_par.timeUnit[7]:
                end_date = end_date.shift(seconds=time_number_this)
            if com_par.timeUnit[6]:
                end_date = end_date.shift(minutes=time_number_this)
            if com_par.timeUnit[5]:
                end_date = end_date.shift(hours=time_number_this)
            if com_par.timeUnit[3]:
                start_date = start_date.replace(hour=0, minute=0, second=0)
                end_date = end_date.shift(days=time_number_this - 1).replace(hour=23, minute=59, second=0)
            if com_par.timeUnit[2]:
                start_date = start_date.replace(day=1, hour=0, minute=0, second=0)
                end_date = end_date.shift(months=time_number_this - 1)
                end_date = end_date.replace(day=end_date.ceil('month').day, hour=23, minute=59, second=0)
            if com_par.timeUnit[1]:
                start_date = start_date.replace(month=1, day=1, hour=0, minute=0, second=0)
                end_date = end_date.shift(years=time_number_this - 1).replace(month=12, day=31, hour=23, minute=59, second=0)
            if com_par.timeUnit[0]:
                century = start_date.year / 100
                start_date = start_date.replace(year=century * 100, month=1, day=1, hour=0, minute=0, second=0)
                end_date = end_date.replace(year=(century + time_number_this) * 100, month=12, day=31, hour=23, minute=59, second=0)
        
        elif(u'来' in time_around_suffix):
            if com_par.timeUnit[1]:
                start_date = start_date.shift(years=-time_number_this + 1).replace(month=1, day=1, hour=0, minute=0, second=0)
                end_date = end_date.replace(month=12, day=31, hour=23, minute=59, second=0)
            
            
        #process the suffix word "后期"
        else:
            if com_par.timeUnit[0]:
                if(time_number_this == 1): #have the "本今"
                    time_number_this = start_date.year / 100 + 1
                start_year = self.century_period[time_around_suffix][0]
                end_year = self.century_period[time_around_suffix][1]
                start_date = start_date.replace(year=(time_number_this - 1) * 100 + start_year, month=1, day=1, hour=0, minute=0, second=0)
                end_date = end_date.replace(year=(time_number_this - 1) * 100 + end_year, month=12, day=31, hour=23, minute=59, second=0)
                
        com_par.startDate = start_date
        com_par.endDate = end_date
        return com_par
        
    def cal_century(self, time_num, time_word, basetime, com_par):
        """
        Function:
            1.process the "20世纪": number + time_word
        Parameters:
            1.time_num: str, [0-9]
            2.time_word: srt, 世纪
            3.basetime: str
            4.com_par: CommonParser
        """
        
        #set the timeUnit
        com_par = self.set_timeUnit(com_par, time_word)
        
        time_num = digitconv.getNumFromHan(time_num)
        start_date = arrow.get(basetime)
        end_date = arrow.get(basetime)
        
        if(com_par.timeUnit[0]):
            com_par.startDate = start_date.replace(year=(time_num - 1) * 100, month=1, day=1, hour=0, minute=0, second=0)
            com_par.endDate = end_date.replace(year=(time_num - 1) * 100 + 99, month=12, day=31, hour=23, minute=59, second=0)
            
        return com_par
        
#    def process_apm(self, time_word, com_par):
#        """
#        Function:
#            process the apm and call the apm_dict
#        Parameter:
#            1.time_word: str,  "中午、夜间、下午、晚上...", apm
#            2.com_par: CommonParser,
#        Return:
#            1.com_par: CommonParser,
#        """
#        pat_date_time = u'([0-9]*).*?-([0-9]*).*?-([0-9]*).*?-([0-9]*).*?:([0-9]*).*?:([0-9]*).*?'
#        apm_time = self.apm_dict[time_word] 
#        
#        year_shift = 0
#        day_shift = 0
#        time_period_split = TimePeriodSplit()
#        
#        #get the start time 
#        mat_date_time = re.match(pat_date_time, apm_time[u'start'])
#        offset = apm_time[u'offset']
#        
#        #offset has "start_1d"
#        if(offset == u'add_1d'):
#            day_shift = 1
#        elif(offset == u'sub_1d'):
#            day_shift = -1
#            
#        start_time, _ = time_period_split.date_setting(mat_date_time, com_par.date, year_shift, day_shift, com_par.timeUnit)
#        
#        #get the end time 
#        mat_date_time = re.match(pat_date_time, apm_time[u'end'])
#        
#        offset = apm_time[u'offset']
#        #offset the year or day
#        if(offset == u'1y'):
#            year_shift = 1
#        if(offset == u'1d'):
#            day_shift = 1
#            
#        end_time, _ = time_period_split.date_setting(mat_date_time, com_par.date, year_shift, day_shift, com_par.timeUnit)
#
#        com_par.startDate = start_time
#        com_par.endDate = end_time
#        
#        return com_par
    
    def call_TimeSlotValue(self, entity, basetime):
        """
        Function:
            1.call the function "TimeSlotValue"
        Parameters:
            1.entity: str,
            2.basetime: str
        """
        ent = Entity()
        ent.entity = entity
        ent.type = "time_point"
        ent.abs_rel = "relative"
        ent.is_refer = False
        ent.is_freq = False
        time_slot_value = TimeSlotValue(ent, '2018-11-01 12:00:00')
        time = time_slot_value.parse()
        return time
        
    
    def around_parse(self, entity, basetime, com_par):
        """
        Function:
            choose which the pattern
        """
        matcher_year = re.match(self.pattern_year, entity)
        matcher_half_double_year = re.match(self.pattern_half_double_year, entity)
        matcher_half = re.match(self.pattern_half, entity)
        matcher_half_since = re.match(self.pattern_half_since, entity)
        matcher_around_num_prefix = re.match(self.pattern_around_num_prefix, entity)        
        matcher_around_num_prefix_age_reach = re.match(self.pattern_around_num_prefix_age_reach, entity)
        matcher_around_num_prefix_age_century_reach = re.match(self.pattern_around_num_prefix_age_century_reach, entity)
        matcher_around_num_prefix_age_suffix_reach = re.match(self.pattern_around_num_prefix_age_suffix_reach, entity)
        matcher_around_num_prefix_age_suffix = re.match(self.pattern_around_num_prefix_age_suffix, entity)
        matcher_around_num_prefix_age2 = re.match(self.pattern_around_num_prefix_age2, entity)
        matcher_around_num_prefix_age = re.match(self.pattern_around_num_prefix_age, entity)
#        matcher_apm = re.match(self.pattern_apm, entity)
        matcher_around_num_suffix = re.match(self.pattern_around_num_suffix, entity)
        matcher_around_suffix = re.match(self.pattern_around_suffix, entity)
        matcher_around_only_time = re.match(self.pattern_around_only_time, entity)
        
        if(matcher_half): #have the "半" number and "前"
            com_par = self.cal_half_time_parse(matcher_half.group(1), matcher_half.group(2), matcher_half.group(3), basetime, com_par)
            if(matcher_year): #have the "2013年下半年"
                year = int(matcher_year.group(1))
                com_par.startDate = com_par.startDate.replace(year=year)
                com_par.endDate = com_par.endDate.replace(year=year)
            elif(matcher_half_double_year): #have the double year "明年上半年"
                time = self.call_TimeSlotValue(entity, basetime)
                matcher_year = re.match(self.pattern_year2, time.startDate)
                year = int(matcher_year.group(1))
                com_par.startDate = com_par.startDate.replace(year=year)
                com_par.endDate = com_par.endDate.replace(year=year)
            
                #have "以来" in "去年下半年以来"
                if(matcher_half_double_year.group(3)):
                    com_par.endDate = arrow.get(basetime)
                
        
        elif(matcher_half_since):
            com_par = self.cal_half_time_parse(matcher_half_since.group(3), matcher_half_since.group(1), matcher_half_since.group(2), basetime, com_par)
            
        elif(matcher_around_num_prefix): #have the num and prefix word. 前1个世纪
            if(matcher_year): # have "2013年前11个月"
                basetime = list(basetime)
                basetime[0:4] = matcher_year.group(1)
                basetime[4:-1] = u'-12-01 00:00:00'
                basetime = ''.join(basetime)
            com_par = self.cal_integer_time_prefix_parse(matcher_around_num_prefix.group(1), matcher_around_num_prefix.group(2), matcher_around_num_prefix.group(3), basetime, com_par)
            if(u'来' in matcher_around_num_prefix.group(4)):
                com_par.endDate = arrow.get(basetime)
            
        elif(matcher_around_num_prefix_age_reach): #have the num and prefix word in "到". 上世纪80年代至上世纪90年代
            start_com = CommonParser(basetime)
            end_com = CommonParser(basetime)
            start_com = self.cal_integer_time_prefix_parse(matcher_around_num_prefix_age_reach.group(1), matcher_around_num_prefix_age_reach.group(3), matcher_around_num_prefix_age_reach.group(4), basetime, start_com)
            end_com = self.cal_integer_time_prefix_parse(matcher_around_num_prefix_age_reach.group(6), matcher_around_num_prefix_age_reach.group(8), matcher_around_num_prefix_age_reach.group(9), basetime, end_com)
            com_par.startDate = start_com.startDate      
            com_par.endDate = end_com.endDate

        elif(matcher_around_num_prefix_age_century_reach or matcher_around_num_prefix_age_suffix_reach): #have the num and prefix word in "到". 上世纪80年代至本世纪初期
            start_com = CommonParser(basetime)
            end_com = CommonParser(basetime)
            if(matcher_around_num_prefix_age_suffix_reach):
                matcher_around_num_prefix_age_century_reach = matcher_around_num_prefix_age_suffix_reach
            start_com = self.cal_integer_time_prefix_parse(matcher_around_num_prefix_age_century_reach.group(1), matcher_around_num_prefix_age_century_reach.group(3), matcher_around_num_prefix_age_century_reach.group(4), basetime, start_com)
            end_com = self.cal_integer_time_suffix_parse(matcher_around_num_prefix_age_century_reach.group(8), matcher_around_num_prefix_age_century_reach.group(6), matcher_around_num_prefix_age_century_reach.group(7), basetime, com_par)
            com_par.startDate = start_com.startDate      
            com_par.endDate = end_com.endDate
        
        elif(matcher_around_num_prefix_age_suffix): #process the "上世纪90年代后期"
            com_par = self.cal_integer_time_prefix_parse(matcher_around_num_prefix_age_suffix.group(1), matcher_around_num_prefix_age_suffix.group(3), matcher_around_num_prefix_age_suffix.group(4) + matcher_around_num_prefix_age_suffix.group(5), basetime, com_par)
        
        elif(matcher_around_num_prefix_age2): #process the "上世纪7、80年代"
            com_par = self.cal_integer_time_prefix_parse(matcher_around_num_prefix_age2.group(1), matcher_around_num_prefix_age2.group(3) + matcher_around_num_prefix_age2.group(4), matcher_around_num_prefix_age2.group(5), basetime, com_par)            
            
        elif(matcher_around_num_prefix_age): #process the "上世纪90年代"
            com_par = self.cal_integer_time_prefix_parse(matcher_around_num_prefix_age.group(1), matcher_around_num_prefix_age.group(3), matcher_around_num_prefix_age.group(4), basetime, com_par)
            
        elif(matcher_around_num_suffix): #process "20世纪后期"、 "1年内"
            if(matcher_year):
                com_par = None
            else:
                com_par = self.cal_integer_time_suffix_parse(matcher_around_num_suffix.group(3), matcher_around_num_suffix.group(1), matcher_around_num_suffix.group(2), basetime, com_par)
            
        elif(matcher_around_suffix): #process the "本世纪初期"
            if(matcher_around_suffix.group(3)):
                com_par = self.cal_integer_time_suffix_parse(matcher_around_suffix.group(3), matcher_around_suffix.group(1), matcher_around_suffix.group(2), basetime, com_par)
            elif(matcher_around_suffix.group(4) == u''): # 全年
                com_par = self.cal_integer_time_suffix_parse("", matcher_around_suffix.group(1), matcher_around_suffix.group(2), basetime, com_par)
            else:
                com_par = None
            
        elif(matcher_around_only_time): #process the "20世纪"
            com_par = self.cal_century(matcher_around_only_time.group(1), matcher_around_only_time.group(2), basetime, com_par)
#        elif(matcher_apm): #noly have the apm time word
#            com_par = self.process_apm(entity, com_par)
            
        else:
            com_par = None
        return com_par
    
if __name__ == "__main__":
    time_period_around = TimePeriodAround()
#    entity = u'1年中'
    entity = u'八年来' #全年 全天 本年度
    basetime = u'2018-11-02 12:00:00'

    com_par = CommonParser(basetime)
    com_par = time_period_around.around_parse(entity, basetime, com_par)
    print("entity:" + entity)
    if(com_par):
        print("result:" + com_par.date.format("YYYY-MM-DD HH:mm:ss"))
        print('start:' + com_par.startDate.format("YYYY-MM-DD HH:mm:ss"))
        print('end:' + com_par.endDate.format("YYYY-MM-DD HH:mm:ss"))
    else:
        print("no result")
    