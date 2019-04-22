# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 17:32:05 2019

Process the season in "time_period" type

@author: leolqli
"""
import re
import arrow
from utils import digitconv

class Season:
    def __init__(self):
        self.pattern_around = u'([0-9]{4})年.*?(前|后)([1-4一二三四]{1,}).*?[季度|季度内]$'
        self.pattern_number_group = u'([0-9]{4})年.*?([1-4一二三四]{1}).*?([1-4一二三四]{1}).*?([1-4一二三四]{0,1}).*?([1-4一二三四]{0,1})[季度|季度内]$'
        self.pattern_origin = u'([0-9]{4})年.*?([1-4一二三四]{1,}).*?[季度|季度内]$'
        self.pattern_no_year_pre_now_around = u'.*?([去|今|明]{1,}).*?(前|后)([0-9零一二三四五六七八九十|下|上|这]{1,}).*?[季度|季度内]$'
        self.pattern_no_year_around = u'.*?(前|后|上|下)([0-9零一二三四五六七八九十|下|上|这]{1,}).*?[季度|季度内]$'
        self.pattern_no_year_previous = u'.*?(去).*?([1-4一二三四]).*?[季度|季度内](来|以来{0,})$'
        self.pattern_no_year_number_group = u'.*?([1-4一二三四]{1}).*?([1-4一二三四]{1}).*?([1-4一二三四]{0,1}).*?([1-4一二三四]{0,1})[季度|季度内]$'
        self.pattern_no_year = u'.*?([1-4一二三四|下|上|这]{1,}).*?(季度|季度内|季度末)$'

        #match the time format
        self.pattern_basetime = u'([\\d]{4})-([\\d]{1,2})-([\\d]{1,2}).*?'
        self.season_dict = {0:(1, 3), 1:(4, 6), 2:(7, 9), 3:(10, 12)}   

    
    def get_dev_up2down(self, basemonth, deviation):
        """
        Function:
            process the sample like "上*", "下*" and "这"
        Parameters:
            1.basemonth: str, current month of basetime
            2.deviation: str, get the value according to the pattern "[1-4一二三四|下|上|这]"
        """
        #get the now season according to base month
        shift_season = 0
        basemonth = int(basemonth)
        for season in self.season_dict.keys():
            if(basemonth >= self.season_dict[season][0] and basemonth <= self.season_dict[season][1]):
                shift_season = season
                break
        
        if u'上' in deviation:
            for i in range(len(deviation)):
                shift_season -= 1
            
        elif u'下' in deviation:
            for i in range(len(deviation)):
                shift_season += 1
                
        elif deviation in u'1-4一二三四':
            shift_season = digitconv.getNumFromHan(deviation) - 1
        
        #get the shift of year and season
        year = shift_season / 4
        shift_season = shift_season % 4
        return year, shift_season
    
    def get_dev_around(self, year, basemonth, season_number, around_text):
        """
        Function:
            entity has the around_text "前|后"
        Parameters:
            1.year: int
            2.basemonth: str, 
            3.season_number: str, '0-9零一二三四五六七八九十' or "上下"
            4.around_text: str, "前|后" or "去"
        Return: start_year, end_year, start_month, end_month
        """
        basemonth = int(basemonth)
        start_month = 0
        end_month = 0
        for season in self.season_dict.keys():
            if(basemonth >= self.season_dict[season][0] and basemonth <= self.season_dict[season][1]):
                start_month = self.season_dict[season][0]
                end_month = self.season_dict[season][1]
                break

        shift = 0
        #if the season_number have the "上下"
        if(u'上' in season_number or u'下' in season_number):
            if(u'上' in season_number):
                shift = - (len(season_number) + 1)
                end_month = start_month - 1
                start_month = start_month + shift * 3  
            
            if(u'下' in season_number):
                shift = len(season_number) + 1
                start_month = (end_month + 1) % 12
                end_month = end_month + shift * 3              
            
        else:
            if(around_text == u'前' or around_text == u'上'):
                shift = - digitconv.getNumFromHan(season_number)
                end_month = start_month - 1
                start_month = start_month + shift * 3
            elif(around_text == u'后' or around_text == u'下'):
                shift = digitconv.getNumFromHan(season_number)
                start_month = (end_month + 1) % 12
                end_month = end_month + shift * 3
            elif(around_text == u'去'):
                shift = digitconv.getNumFromHan(season_number) - 1
                start_month = self.season_dict[shift][0]
                end_month = self.season_dict[shift][1]
                year = year - 1
                return year, year, start_month, end_month

        #carry and abdicate the year
        start_year = year
        end_year = year
        if(start_month <= 0 and end_month >= 0):
            start_month += -1
            start_year = year - (abs(start_month) / 12 + 1)
            end_year = year
            start_month = 12 - abs(start_month + 1) % 12
        elif(start_month >= 0 and end_month >= 12):
            start_year = year + (end_month / 12)
            end_year = year + (end_month / 12)
            end_month = end_month % 12
        
        #when the end_month=12, the value of it will be 0. so reset the value of end_month
        if(end_month == 0):
            end_month = 12
            end_year -= 1 
            
        return start_year, end_year, start_month, end_month
    
    def get_dev_aroundlist(self, year, season_number, around_text):
        """
        Function:
            entity has the around_text "去|今|明" "前|后"
        Parameters:
            1.year: int
            2.season_number: str, '0-9零一二三四五六七八九十'
            3.around_text: list, ["去|今", "前|后"] 
        Return: start_year, end_year, start_month, end_month
        """
        start_month = 0
        end_month = 0
        
        start_year = 0
        end_year = 0
        if(around_text[0] == u'去'):
            start_year = year - 1
            end_year = year - 1
        elif(around_text[0] == u'今'):
            start_year = year
            end_year = year
        elif(around_text[0] == u'明'):
            start_year = year + 1
            end_year = year + 1
            
        shift = digitconv.getNumFromHan(season_number) - 1
        if(around_text[1] == u'前'):
            start_month = 1
            end_month = 3 + 3 * shift
        elif(around_text[1] == u'后'):
            start_month = 10 - 3 * shift
            end_month = 12
        
        return start_year, end_year, start_month, end_month
    
    def get_dev_numlist(self, numberlist):
        """
        Function:
            entity has the number list "一、二、三...."
        Parameters:
            1.numberlist: list, '0-9零一二三四五六七八九十'
        Return: start_month, end_month
        """
        season_start = digitconv.getNumFromHan(numberlist[0]) - 1
        season_end = digitconv.getNumFromHan(numberlist[-1]) - 1
        start_month = self.season_dict[season_start][0]
        end_month = self.season_dict[season_end][1]
        
        return start_month, end_month
        
    def season_period_dev(self, basetime, season_number, around_text):
        """
        Function:
            process the season "季度", get the deviation of the season
        Parameters::
            1.basetime: str, year time or year-month-day time
            2.season_number: str, '[1-4一二三四|下|上]'; list, ['一', '二', '三'...]
            3.around_text: str, '前|后' or '去'; list, ['去|今', '前|后']
        Return: start_year, end_year, start_month, end_month
        """
        #process the word '下' and '上'
        shift_season = 0
        year = 0
        
        #according to the pattern_basetime, no have the year in entity
        matcher = re.match(self.pattern_basetime, basetime)
        basemonth = ''
        if(matcher):
            basemonth = matcher.group(2)
            
            #have the word "去|今" and "前|后"
            if(type(around_text) == list):
                year = arrow.get(basetime).year
                start_year, end_year, start_month, end_month = self.get_dev_aroundlist(year, season_number, around_text)
                return start_year, end_year, start_month, end_month
                
            #have the word "前|后"
            if(around_text == u'前' or around_text == u'后' or around_text == u'上' or around_text == u'下'):
                year = arrow.get(basetime).year 
                start_year, end_year, start_month, end_month = self.get_dev_around(year, basemonth, season_number, around_text)
                return start_year, end_year, start_month, end_month
            
            #have the word "去"
            elif(around_text == u'去'):
                year = arrow.get(basetime).year 
                start_year, end_year, start_month, end_month = self.get_dev_around(year, basemonth, season_number, around_text)
                return start_year, end_year, start_month, end_month
                
            elif(type(season_number) == list):
                start_month, end_month = self.get_dev_numlist(season_number)
                year = arrow.get(basetime).year
                return year, year, start_month, end_month
            else:
                #have the "上|下"
                year, shift_season = self.get_dev_up2down(basemonth, season_number)
                year = arrow.get(basetime).year + year 
                
        #process the word '前|后 1-4一二三四', have the year in entity
        else:
            if(around_text == u'前' or around_text == u'后'):
                #the start_month = 1, and change the end_month
                if(around_text == u'前'):
                    around_text = u'后'
                    basemonth = '12'
                #the end_month = 12, and change the start_month
                else:
                    around_text = u'前'
                    basemonth = '1'
                _, _, start_month, end_month = self.get_dev_around(year, basemonth, season_number, around_text)
                year = int(basetime)       
                return year, year, start_month, end_month
            
            elif(type(season_number) == list):
                #have "一、二、三..." in entity
                start_month, end_month = self.get_dev_numlist(season_number)
                year = int(basetime)
                return year, year, start_month, end_month
                
            #have the year in entity, no have the "前|后", have the "上|下"
            else:
                year, shift_season = self.get_dev_up2down('1', season_number)
                year = int(basetime)
                
        start_month = self.season_dict[shift_season][0]
        end_month = self.season_dict[shift_season][1]      
        return year, year, start_month, end_month
    
    def season_end(self, basetime, season_number, season_word):
        """
        Function:
            process the "季度末"
        Parameter:
            1.basetime: str
            2.season_number: str, '[1-4一二三四]';
            3.season_word: str, "季度末"
        """
        year = arrow.get(basetime).year
        if(season_word == u'季度末'):
            
            season_number = digitconv.getNumFromHan(season_number)
            
            start_month = self.season_dict[season_number - 1][1]
            end_month = self.season_dict[season_number - 1][1]
                
        return year, year, start_month, end_month
    
    def season_recognise(self, entity, basetime):
        """
        function:
            Recoginse the entity in pattern.
        Return: 
            the start_year, end_year, start_month, end_month, basetime 
        """
        error = False
        start_year = 0
        end_year = 0 
        start_month = 0 
        end_month = 0
        
        #have the basetime in entity, sample like "pattern_around"
        matcher_around = re.match(self.pattern_around, entity)
        #process the sample like 'pattern_origin', have "前|后"  
        matcher_number_group = re.match(self.pattern_number_group, entity)
        #process the sample like 'pattern_origin', have "前|后"            
        matcher_origin = re.match(self.pattern_origin, entity)
        #process the sample like 'pattern_no_year_pre_now_around', have "去|今" and "前|后"
        matcher_no_year_pre_now_around = re.match(self.pattern_no_year_pre_now_around, entity)
        #process the sample like 'pattern_no_year_around', have "前|后"
        matcher_no_year_around = re.match(self.pattern_no_year_around, entity)
        #process the sample like 'pattern_no_year_previous'
        matcher_no_year_previous = re.match(self.pattern_no_year_previous, entity)
        #process the sample like 'pattern_no_year_number_group', have "一、二、三...."
        matcher_no_year_number_group = re.match(self.pattern_no_year_number_group, entity)
        matcher_no_year = re.match(self.pattern_no_year, entity)        
        
        #process the sample like 'pattern_around'
        if(matcher_around):
            start_year, end_year, start_month, end_month = self.season_period_dev(matcher_around.group(1), matcher_around.group(3), matcher_around.group(2))
            basetime = matcher_origin.group(1)
        
        elif(matcher_number_group):
            season_numlist = []
            for i in range(4):
                if(matcher_number_group.group(i+2) != ''):
                    season_numlist.append(matcher_number_group.group(i+2))
            start_year, end_year, start_month, end_month = self.season_period_dev(matcher_number_group.group(1), season_numlist, '')
            basetime = matcher_origin.group(1)            
            
        elif(matcher_origin):
            start_year, end_year, start_month, end_month = self.season_period_dev(matcher_origin.group(1), matcher_origin.group(2), '')
            basetime = matcher_origin.group(1)       
        
        elif(matcher_no_year_pre_now_around):
            around_text = [matcher_no_year_pre_now_around.group(1), matcher_no_year_pre_now_around.group(2)]
            start_year, end_year, start_month, end_month = self.season_period_dev(basetime, matcher_no_year_pre_now_around.group(3), around_text)
        
        elif(matcher_no_year_around):
            #have "前|后"
            start_year, end_year, start_month, end_month = self.season_period_dev(basetime, matcher_no_year_around.group(2), matcher_no_year_around.group(1))
        
        elif(matcher_no_year_previous): #去年4季度
            start_year, end_year, start_month, end_month = self.season_period_dev(basetime, matcher_no_year_previous.group(2), matcher_no_year_previous.group(1))
            if(matcher_no_year_previous.group(3)): #have "以来"
                matcher_basetime = re.match(self.pattern_basetime, basetime)
                end_year = int(matcher_basetime.group(1))
                end_month = int(matcher_basetime.group(2))
            
        elif(matcher_no_year_number_group):
            season_numlist = []
            for i in range(4):
                if(matcher_no_year_number_group.group(i+1) != ''):
                    season_numlist.append(matcher_no_year_number_group.group(i+1))
            start_year, end_year, start_month, end_month = self.season_period_dev(basetime, season_numlist, '')            
        
        elif(matcher_no_year):
            if(matcher_no_year.group(2) == u'季度末'):
                start_year, end_year, start_month, end_month = self.season_end(basetime, matcher_no_year.group(1), matcher_no_year.group(2))
            else:
                start_year, end_year, start_month, end_month = self.season_period_dev(basetime, matcher_no_year.group(1), '')
            
        else:
            print("-----------------------------")
            print("no matcher")
            error = True
                        
        return start_year, end_year, start_month, end_month, basetime, error
        

if __name__ == '__main__':
    season = Season()
#    entity = u'一、二季度'
    entity = u'去年四季度'
    basetime = u'2018-11-01 12:00:00'
#    basetime = u'2019'
    start_year, end_year, start_month, end_month, basetime, error = season.season_recognise(entity, basetime)
