# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:45:01 2019

@author: leolqli
@function: split the "time_split" type, and process different conditions
"""
import re
import json
from time_Info.year import Year
from time_Info.month import Month
from time_Info.day import Day
from time_Info.apm import Apm
from time_Info.hour import Hour
from time_Info.minute import Minute
from time_Info.second import Second
from time_Info.lunarMonth import lunarMonth
from time_Info.lunarDay import lunarDay
from CommonParser import CommonParser
from TimePeriodProcess import TimePeriodProcess
from TimePeriodAround import TimePeriodAround

import arrow
import pandas as pd

class TimePeriodSplit:
    def __init__(self):
        self.pattern_week = u'.*?(星期|礼拜|周|工作日|双休日).*?'
        self.pattern_season = u'.*?(季度).*?'
        self.pattern_section = u'(.*?)(到|至|和|-|~|—|～|及|－)(.*)'
        self.pattern_section_two = u'(.*?)(到|至|和|-|~|—|～|及)(.*)(到|至|和|-|~|—|～|及)(.*?)'
        self.pattern_since = u'.*?(?!未来)(来|以来)'
        
        #load the apm word
        file_apm = open("./resources/time_duration_apm.json", "r")
        self.apm_dict = json.load(file_apm)
        apm_list = list(self.apm_dict.keys())
        apm_list.sort(key = lambda i: len(i), reverse=True)
        self.pattern_apm = u'(.*?)(?!月初[0-9零一二三四五六七八九两]|年初[0-9零一二三四五六七八九两])(' + '|'.join(apm_list) + u')(.*)'
        
        self.pattern_basetime = u'.*?([\\d]{4,}).*?'
        
        
    def date_setting(self, mat_date_time, date_time, year_shift, day_shift, timeUnit):
        """
        Function:
            setting the date according to the apm
        Parameter:
            1.mat_date_time: re.match,  "中午、夜间、下午、晚上"
            2.date_time: allow,
            3.year_shift: int, offset the year or day in end_date
            4.day_shift: int, offset the day or day in end_date
            5.timeUnit: list, label which position should be changed
        Return:
            1.date_time: allow, return the modified time
            2.timeUnit: list, label which position should be changed 
        """
        if(mat_date_time.group(1)):
            replace_year = str(date_time.year)[0: (4 - len(mat_date_time.group(1)))]
            replace_year += mat_date_time.group(1)
            date_time = date_time.replace(year=int(replace_year))
            timeUnit[1] = True
        if(mat_date_time.group(2)):
            date_time = date_time.replace(month=int(mat_date_time.group(2)))
            timeUnit[2] = True
        if(mat_date_time.group(3)):
            replace_day = int(mat_date_time.group(3))
            #if the replace_day > the last day of the month
            if(replace_day > date_time.ceil('month').day):
                date_time = date_time.replace(day=date_time.ceil('month').day)
            else:
                date_time = date_time.replace(day=replace_day)
            timeUnit[3] = True
        if(mat_date_time.group(4)):
            date_time = date_time.replace(hour=int(mat_date_time.group(4)))
            timeUnit[4] = True
        if(mat_date_time.group(5)):
            date_time = date_time.replace(minute=int(mat_date_time.group(5)))
            timeUnit[5] = True
        if(mat_date_time.group(6)):
            date_time = date_time.replace(second=int(mat_date_time.group(6)))
            timeUnit[6] = True
        
        #have offset in year and day
        if(year_shift != 0):
            date_time = date_time.replace(year=date_time.year)
            date_time = date_time.shift(years=year_shift)
        if(day_shift != 0):
            date_time = date_time.replace(day=date_time.day)
            date_time = date_time.shift(days=day_shift)
            
        return date_time, timeUnit
        
    def process_apm(self, apm, date_time, is_start, timeUnit):
        """
        Function:
            process the apm and call the apm_dict
        Parameter:
            1.apm: str,  "中午、夜间、下午、晚上..."
            2.date_time: allow,
            3.is_start: boolean, True is the "start time" and False is the "end time"
            4.timeUnit: list, label which position should be changed
        Return:
            1.date_time: allow, return the modified time
            2.timeUnit: list, label which position should be changed 
        """
        pat_date_time = u'([0-9]*).*?-([0-9]*).*?-([0-9]*).*?-([0-9]*).*?:([0-9]*).*?:([0-9]*).*?'
        apm_time = self.apm_dict[apm] 
        
        year_shift = 0
        day_shift = 0
        if(is_start):
            mat_date_time = re.match(pat_date_time, apm_time[u'start'])
            offset = apm_time[u'offset']
            
            #offset has "start_1d"
            if(offset == u'add_1d'):
                day_shift = 1
            elif(offset == u'sub_1d'):
                day_shift = -1
                
            date_time, timeUnit = self.date_setting(mat_date_time, date_time, year_shift, day_shift, timeUnit)
        else:
            mat_date_time = re.match(pat_date_time, apm_time[u'end'])
            
            offset = apm_time[u'offset']
            #offset the year or day
            if(offset == u'1y'):
                year_shift = 1
            if(offset == u'1d' or offset == u'add_1d'):
                day_shift = 1
                
            date_time, timeUnit = self.date_setting(mat_date_time, date_time, year_shift, day_shift, timeUnit)
            
        return date_time, timeUnit
            
        
    def process_time_period(self, time, apm, basetime, is_start, com_par):
        """
        Function:
            process the time_period and get the date time.
        Parameter:
            1.time: str, "年月日时分秒"
            2.apm: str,  "中午、夜间、下午、晚上"
            3.basetime: str,
            4.is_start: boolean, True is the "start time" and False is the "end time"
            5.com_par: CommonParse, store the startDate and endDate
        Return:
            1.date_time: arrow, modified time
            2.timeUnit: list, label which position should be changed 
        """
        #check if have the basetime in "time"
        matcher_basetime = re.match(self.pattern_basetime, time)
        if(matcher_basetime):
            basetime = matcher_basetime.group(1) + u'-01-01'
        
        #process the "年、月、日、apm、时、分、秒"
        com_par.timeUnit = [False,False,False,False,False,False,False,False]
        com_par.date = arrow.get(basetime)
        #apm_flag = True: call the Apm()
        apm_flag = False  
        if not com_par.timeUnit[1]:
            year_proc = Year()
            year_proc.get_year(time, basetime, com_par)
        if not com_par.timeUnit[2]:
            month_proc = Month()
            month_proc.get_month(time, basetime, com_par)
        if not com_par.timeUnit[2]:
            month_proc_lun = lunarMonth()
            month_proc_lun.get_month(time, basetime, com_par)
        if not com_par.timeUnit[3]:
            day_proc = Day()
            day_proc.get_day(time, basetime, com_par)
        if not com_par.timeUnit[3]:
            day_proc_lun = lunarDay()
            day_proc_lun.get_day(time, basetime, com_par)
        if not com_par.timeUnit[4]:
            #has apm and the tiem 
            apm_proc = Apm()
            apm_flag = apm_proc.judge_apm_hour(time, com_par)
            #only has apm words
            if(apm != ''):
                com_par.timeUnit[4] = True
        if not com_par.timeUnit[5]:
            hour_proc = Hour()
            hour_proc.get_hour(time, basetime, com_par) 
        if not com_par.timeUnit[6]:
            minute_proc = Minute()
            minute_proc.get_minute(time, basetime, com_par)
        if not com_par.timeUnit[7]:
            second_proc = Second()
            second_proc.get_second(time, basetime, com_par)
        
        date_time = arrow.get(basetime)
        no_time = True
        if(is_start):
            #process the start date
            if com_par.timeUnit[7]:
                date_time = com_par.date
                no_time = False
            elif com_par.timeUnit[6]:
                date_time = com_par.date.replace(second=0)
                no_time = False
            elif com_par.timeUnit[5]:
                date_time = com_par.date.replace(minute=0).replace(second=0)
                no_time = False
            elif com_par.timeUnit[3]:
                date_time = com_par.date.replace(hour=0).replace(minute=0).replace(second=0)
                no_time = False
            elif com_par.timeUnit[2]:
                date_time = com_par.date.replace(day=1).replace(hour=0).replace(minute=0).replace(second=0)
                no_time = False
            elif com_par.timeUnit[1]:
                date_time = com_par.date.replace(month=1).replace(day=1).replace(hour=0).replace(minute=0).replace(second=0)
                no_time = False
        
        else:
            #process the end date
            if com_par.timeUnit[7]:
                date_time = com_par.date
                no_time = False
            elif com_par.timeUnit[6]:
                date_time = com_par.date.replace(second=59)
                no_time = False
            elif com_par.timeUnit[5]:
                date_time = com_par.date.replace(minute=59).replace(second=59)
                no_time = False
            elif com_par.timeUnit[3]:
                date_time = com_par.date.replace(hour=23).replace(minute=59).replace(second=59)
                no_time = False
            elif com_par.timeUnit[2]:
                date_time = com_par.date.replace(day=com_par.date.ceil('month').day).replace(hour=23).replace(minute=59).replace(second=59)
                no_time = False
            elif com_par.timeUnit[1]:
                date_time = com_par.date.replace(month=12).replace(day=com_par.date.ceil('month').day).replace(hour=23).replace(minute=59).replace(second=59)
                no_time = False

        #set the apm time
        if(com_par.timeUnit[4] and not apm_flag):
            date_time, com_par.timeUnit = self.process_apm(apm, date_time, is_start, com_par.timeUnit)
            no_time = False
        #set basetime in commonParse.date
        com_par.date = arrow.get(basetime)
        
        if(no_time):
            return None, com_par.timeUnit
        
        return date_time, com_par.timeUnit
        
    def suffix_align_prefix(self, start_timeUnit, end_timeUnit, start_date, end_date, suffix_time, apm, basetime, com_par):
        """
        Function:
            1.ensure start_timeUnit != end_timeUnit
            2.make the suffix time align to the prefix time
        Parameters:
            1.start_timeUnit: list, the length of it is 7. "#世纪、年、月、日、apm、时、分、秒"
            2.end_timeUnit: list, the length of it is 7. "#世纪、年、月、日、apm、时、分、秒"
            3.start_date: arrow
            4.end_date: arrow
            5.suffix_time: str, the suffix time 
        """
        
        #ensure start_timeUnit != end_timeUnit
        if(start_timeUnit == end_timeUnit):
            return end_date
        
        #get the true position in end_timeUnit
        end_index_true = -1
        while(end_index_true < len(end_timeUnit)):
            if(end_timeUnit[end_index_true] == True):
                break
            end_index_true += 1
            
        if(1 < end_index_true):
            end_date = end_date.replace(year = start_date.year)
        if(2 < end_index_true):
            end_date = end_date.replace(month = start_date.month)            
        if(3 < end_index_true):
            end_date = end_date.replace(day = start_date.day)
        if(4 < end_index_true):
            #process the apm
            suffix_time = end_date.format("YYYY-MM-DD").encode("utf-8") + u' ' + apm + suffix_time
            end_date, com_par.timeUnit = self.process_time_period(apm + suffix_time, apm, basetime, False, com_par)
            
            if(com_par.timeAPMInfo != None):
                suffix_time = end_date.format("YYYY-MM-DD").encode("utf-8") + u' ' + suffix_time
                end_date, com_par.timeUnit = self.process_time_period(suffix_time, '', basetime, False, com_par)
                
        if(5 < end_index_true):
            end_date = end_date.replace(hour = start_date.hour)
        if(6 < end_index_true):
            end_date = end_date.replace(minute = start_date.minute)
        if(7 < end_index_true):
            end_date = end_date.replace(second = start_date.second)
            
        return end_date
        
    def prefix_align_suffix(self, start_timeUnit, end_timeUnit, start_date, end_date, prefix_time, apm, basetime, com_par):
        """
        Function:
            1.make the prefix time align to the suffix time
        Parameters:
            1.start_timeUnit: list, the length of it is 7. "#世纪、年、月、日、apm、时、分、秒"
            2.end_timeUnit: list, the length of it is 7. "#世纪、年、月、日、apm、时、分、秒"
            3.start_date: arrow
            4.end_date: arrow
            5.prefix_time: str, the suffix time 
            6.apm: str,
            7.basetime: str,
            8.com_par:  CommonParse
        """
        #get the last true position in start_timeUnit
        start_last_true = 0
        for i, has_true in enumerate(start_timeUnit):
            if(has_true == True):
                start_last_true = i + 1
            
        
        #get the true position in end_timeUnit
        end_index_true = 0
        for i, has_true in enumerate(end_timeUnit):
            if(has_true == True):
                end_index_true = i
            
        #if the start_last_true >= end_index_true, this means that no have align
        if(start_last_true > end_index_true):
            return start_date
        
        if(1 == end_index_true):
            start_date, com_par.timeUnit = self.process_time_period(prefix_time + u"年", '', basetime, True, com_par)
        if(2 == end_index_true):
            start_date, com_par.timeUnit = self.process_time_period(prefix_time + u"月", '', basetime, True, com_par)
        if(3 == end_index_true):
            start_date, com_par.timeUnit = self.process_time_period(prefix_time + u"日", '', basetime, True, com_par)
#        if(4 == end_index_true):
#            #process the apm 
#            start_date, com_par.timeUnit = self.process_time_period(apm + prefix_time, apm, basetime, False, com_par)
#            
#            if(com_par.timeAPMInfo != None):
#                start_date, com_par.timeUnit = self.process_time_period(prefix_time, '', basetime, False, com_par)
                
        if(5 == end_index_true):
            start_date, com_par.timeUnit = self.process_time_period(prefix_time + u"时", '', basetime, True, com_par)
        if(6 == end_index_true):
            start_date, com_par.timeUnit = self.process_time_period(prefix_time + u"分", '', basetime, True, com_par)
        if(7 == end_index_true):
            start_date, com_par.timeUnit = self.process_time_period(prefix_time + u"秒", '', basetime, True, com_par)
            
        return start_date
    
    def split_entity(self, entity, basetime, com_par):
        """
        Function:
            1.split the "...到|至...", and get two time
        Parameter:
            1.entity: str, The time entity is processed.
            2.basetime: str, 
            3.com_par: CommonParse, store the startDate and endDate
        Return:
            1.com_par: CommonParse,
            2.None: represent no matcher
        """

        matcher_section = re.match(self.pattern_section, entity)
        matcher_section_two = re.match(self.pattern_section_two, entity)
        
        #have two section, and is error
        if(matcher_section_two):
            if(matcher_section_two.group(2) != matcher_section_two.group(4)):
                return None
        
        #split the entity into to prefix_time and suffix_time
        if(matcher_section):
            prefix_time = matcher_section.group(1) 
            suffix_time = matcher_section.group(3)
            
            #"至今" + "天"
            if(suffix_time == u'今'):
                suffix_time += u'天'
            
            apm = ''
            
            #split the time and apm in prefix_time
            matcher_apm_prefix = re.match(self.pattern_apm, prefix_time)
            is_start = True
            if(matcher_apm_prefix and matcher_apm_prefix.group):
                apm = matcher_apm_prefix.group(2)
#                prefix_time = matcher_apm_prefix.group(1) + matcher_apm_prefix.group(3)
                date_time, start_timeUnit = self.process_time_period(prefix_time, matcher_apm_prefix.group(2), basetime, is_start, com_par)
            else:
                date_time, start_timeUnit = self.process_time_period(prefix_time, '', basetime, is_start, com_par)
            
            com_par.startDate = date_time
            
            #split the time and apm in suffix_time
            matcher_apm_suffix = re.match(self.pattern_apm, suffix_time)
            is_start = False
            if(matcher_apm_suffix):
                apm = matcher_apm_suffix.group(2)
#                suffix_time = matcher_apm_suffix.group(1) + matcher_apm_suffix.group(3)
                date_time, end_timeUnit = self.process_time_period(suffix_time, matcher_apm_suffix.group(2), basetime, is_start, com_par)
            else:
                date_time, end_timeUnit = self.process_time_period(suffix_time, '', basetime, is_start, com_par)
                
            com_par.endDate = date_time
            
            
            #Supplemental suffix and prefix time
            if(suffix_time == u'今天'):
                return com_par
            elif(com_par.endDate):
                #make the prefix time align to the suffix time
                com_par.startDate = self.prefix_align_suffix(start_timeUnit, end_timeUnit, com_par.startDate, com_par.endDate, prefix_time, apm, basetime, com_par)

                #make the suffix time align to the prefix time
                if(com_par.startDate):
                    com_par.endDate = self.suffix_align_prefix(start_timeUnit, end_timeUnit, com_par.startDate, com_par.endDate, suffix_time, apm, basetime, com_par)
                else:
                    #no startDate
                    return None
            else:
                #no endDate
                return None
            
            return com_par
        else:
            return None

    
    def choose_match(self, entity, basetime, com_par):
        """
        Function:
            2.split the "...到|至...", and get two time
        Parameter:
            1.entity: str, The time entity is processed.
            2.basetime: str, 
            3.com_par: CommonParse, store the startDate and endDate
        Return:
            1.com_par: CommonParse,
            2.None: represent no matcher
        """
        time_period_process = TimePeriodProcess()
        
        #process the season type
        matcher_season = re.match(self.pattern_season, entity)
        #process the week type
        matcher_week = re.match(self.pattern_week, entity)
        matcher_section = re.match(self.pattern_section, entity)
        matcher_since = re.match(self.pattern_since, entity)
        
        #season
        if(matcher_season):
            com_par, error = time_period_process.season_each_entity(entity, basetime, com_par)
            if(error):
                com_par = None        
        #week
        elif(matcher_week):
            com_par, error = time_period_process.week_each_entity(entity, basetime, com_par)
            if(error):
                com_par = None
                
        #process the not week type ans no season type
        if(com_par == None or (matcher_week == None and matcher_season == None)):
            #process the "前后"、"世纪" time 
            com_par = CommonParser(basetime)
            time_period_around = TimePeriodAround()
            com_par = time_period_around.around_parse(entity, basetime, com_par)
            
            if(com_par == None):
                com_par = CommonParser(basetime)
                
                if(matcher_section): #have the "到"
                    com_par = self.split_entity(entity, basetime, com_par)
                
                elif(matcher_since): #have the "以来"
                    entity = entity + u"到" + entity
                    com_par = self.split_entity(entity, basetime, com_par)
                    if(com_par):
                        com_par.endDate = arrow.get(basetime)
                    
                else:
                    #no have the "到"
                    entity = entity + u"到" + entity
                    com_par = self.split_entity(entity, basetime, com_par)
        
        if(com_par == None or com_par.timeFormatInfo != ''):
            com_par = None
            print("time format error")
        return com_par
        
    def two_time_compare(self, start_date, end_date):
        """
        Function:
            Judge the size of two times
        Return:
            1.True: start_date <= end_date; False: start_date > end_date
        """
        com = end_date.timestamp - start_date.timestamp
        if(com >= 0):
            return True
        else:
            print("Error: the end_date < start_date")
            return False
                
if __name__ == "__main__":
    time_period_split = TimePeriodSplit()
    entity = u'今年年末'
    basetime = u'2018-11-02 12:00:00'
    
    com_par = CommonParser(basetime)
    com_par = time_period_split.choose_match(entity, basetime, com_par)
    
    print("entity:" + entity)
    if(com_par):
        if(time_period_split.two_time_compare(com_par.startDate, com_par.endDate)):
            print("result:" + com_par.date.format("YYYY-MM-DD HH:mm:ss"))
            print('start:' + com_par.startDate.format("YYYY-MM-DD HH:mm:ss"))
            print('end:' + com_par.endDate.format("YYYY-MM-DD HH:mm:ss"))
        else:
            print("no result")

    else:
        #print no matcher
        print("no result")
        
    

#    file_dataframe = pd.read_csv("./leolqli_data/time_period_finalcase50.csv", encoding='utf-8')
#    
#    line_num = 0
#    dataframe_list = []
#    for idx, data in file_dataframe.iterrows():
#        each_line = []
#        com_par = CommonParser(basetime)
#        com_par = time_period_split.choose_match(data[u'entity'], basetime, com_par)
#
#        print("-----------------------------")
#        print("entity:" + data[u'entity'])
#        each_line.append(data[u'entity'])
#
#        if(com_par):
#            if(time_period_split.two_time_compare(com_par.startDate, com_par.endDate)):
#                print("result:" + com_par.date.format("YYYY-MM-DD HH:mm:ss"))
#                print('start:' + com_par.startDate.format("YYYY-MM-DD HH:mm:ss"))
#                print('end:' + com_par.endDate.format("YYYY-MM-DD HH:mm:ss"))
#                each_line.append(com_par.date.format("YYYY-MM-DD HH:mm:ss"))
#                each_line.append(com_par.startDate.format("YYYY-MM-DD HH:mm:ss"))
#                each_line.append(com_par.endDate.format("YYYY-MM-DD HH:mm:ss"))
#            else:
#                print("no result")
#                each_line.append('')
#                each_line.append('')
#                each_line.append('')
#        else:
#            #print no matcher
#            print("no result")
#            each_line.append('')
#            each_line.append('')
#            each_line.append('')
#            
#        dataframe_list.append(each_line)
#        
#    save_data = pd.DataFrame(dataframe_list, columns=['entity', 'basetime', 'start', 'end'])
#    save_data.to_csv("./leolqli_data/time_period_finalcase50_predict.csv", encoding='utf_8_sig')
    