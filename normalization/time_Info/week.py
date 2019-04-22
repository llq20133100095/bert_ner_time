# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 20:04:05 2018

Process the week in "time_period" type

@author: leolqli
"""
import re
import arrow
from utils import digitconv

class Week:
    def __init__(self):
        self.pattern_origin = u'.*?([上|下|这|全|本|当|前|未来|后]{1,}).*?([0-9零一二三四五六七八九两]{0,}).*?([星期|礼拜|周|工作日|双]{1,})([末|日|六日|休日]{0,})(.*)'
        self.pattern_reach = u'.*?([上|下|这|全|本|当]{0,}).*?(星期|礼拜|周)([1-7零一二三四五六两日天])([到|至]{1,}).*?([上|下|这|全|本|当]{0,}).*?(星期|礼拜|周)([1-7零一二三四五六两日天])$'
        self.pattern_last = u'.*?(?<![上|下|这|全|本|当|前|未来|后])(星期|礼拜|周|双)(?!六|日)([末|内|六日|休日]{1,})'
        self.pattern_num_workday = u'.*?(?<![上|下|这|全|本|当|前|未来|后])([0-9]{1,}).*?(工作日)(内)'
        self.pattern_workday = u'.*?(?<![上|下|这|全|本|当|前|未来|后])(工作日)'
        
    def get_dev_up2down(self, deviation):
        val = 0
        if u'上' in deviation:
            val = (-1) * 7 * len(deviation)
        elif u'下' in deviation:
            val = 7 * len(deviation)
        elif u'前' in deviation:
            val = (-1) * 7
        elif u'未来' in deviation or u'后' in deviation:
            val = 7
        return val
    
    def week_period_dev(self, deviation, week_last, week_number, basetime):
        """
        Function:
            process the week "星期|礼拜", get the deviation of the week
        Parameters::
            1.deviation: str, u'上*', u'下*',u'前面', u'未来' or list ['上*', u'下*']
            2.week_last: str, u'末', u'六日', u'内', u'工作日'
            3.basetime: str
            4.week_number: list, ('一~日', '一~日') or str, "0-9零一二三四五六七八九"
        Return: the deviations of the start time and end time
            "start_dev", "end_dev"
        """
        week_last_num = 0
        workday_num = 0
        if week_last == u'末' or week_last == u'六日' or week_last == u'休日':
            week_last_num = 5
        elif(week_last == u'工作日'):
            workday_num = -2
        
        #get the day of week
        week_day = arrow.get(basetime).weekday()
        
        #get the deviation
        start_val = 0
        end_val = 0
        val = 0
        if(type(deviation) == list):
            start_val = self.get_dev_up2down(deviation[0])
            end_val = self.get_dev_up2down(deviation[1])
        else:
            val = self.get_dev_up2down(deviation)
        
        if(type(week_number) == list):
            #process the sample that has "到|至"
            reach_start = digitconv.getNumFromHan(week_number[0])
            reach_end = digitconv.getNumFromHan(week_number[1])
            start_dev = start_val + (reach_start - week_day - 1)
            end_dev = end_val + (reach_end - week_day - 1)           
            
        #process the “patterm_front_feature”, has "0-9零一二三四五六七八九"
        elif(type(week_number) == type(u'')):
            #constrate the long string in deviation
            mul_week = digitconv.getNumFromHan(week_number)
            if u'前' in deviation or u'上' in deviation or u'这' in deviation:
                start_dev = val * mul_week - week_day
                end_dev = val - week_day + 6
                
                #has the week_last
                if(week_last == u'末' or week_last == u'六日'):
                    start_dev += week_last_num
                    end_dev = start_dev + 1 
                    
                if(week_last == u'工作日'):
                    end_dev = start_dev + 5
                
            elif u'未来' in deviation or u'后' in deviation or u'下' in deviation:
                start_dev = val - week_day
                end_dev = val * mul_week - week_day + 6
                
                #has the week_last
                if(week_last == u'末' or week_last == u'六日'):
                    start_dev = end_dev - 1
                    
                if(week_last == u'工作日'):
                    end_dev = end_dev - 2
                    start_dev = end_dev - week_last_num
                

        else:
            #process the sample that no have "到|至". The deviations of the start time and end time
            start_dev = val - week_day + week_last_num
            end_dev = val - week_day + 6 + workday_num
        
        return start_dev, end_dev
    
    def workday_dev(self, week_number, week_name, week_suffix):
        """
        Function:
            process the time has "工作日"
        Parameters:
            1.week_number: str, '0-9'
            2.week_name: str, '工作日'
            3.week_suffix: str, "内"
        """
        week_number = digitconv.getNumFromHan(week_number)
        
        if(u"内" in week_suffix):
            week = week_number / 5
            day = week_number % 5
            
            start_dev = 1
            end_dev = week * 7 + day
        
        return start_dev, end_dev
        
    def week_recognise(self, entity, basetime):
        """
        function:
            Recoginse the entity in pattern
        Return:
            1.the deviations of the start time and end time
            2.error: if no the match, return the True
        """
        start_dev = 0
        end_dev = 0
        error = False
        #match the pattern in entity, has "到|至"
        matcher = re.match(self.pattern_reach, entity)
        
        #process the sample like 'pattern_reach'
        if(matcher):
            start_dev, end_dev = self.week_period_dev([matcher.group(1), matcher.group(5)], '', \
                [matcher.group(3), matcher.group(7)], basetime)
            
        #process the sample like "pattern_origin"
        else:
            #match the pattern in entity
            matcher = re.match(self.pattern_origin, entity)
            if(matcher):
                #have the "deviation" and "week_last"：下周末
                if(not matcher.group(2) and matcher.group(4) and matcher.group(5) == u''):
                    start_dev, end_dev = self.week_period_dev(matcher.group(1), matcher.group(4), '', basetime)
                
                #have the "deviation" and "number" [0-9零一二三四五六七八九]：下2周
                elif(matcher.group(2) and not matcher.group(4) and matcher.group(3) != u'工作日' and matcher.group(5) == u''):
                    start_dev, end_dev = self.week_period_dev(matcher.group(1), u'', matcher.group(2), basetime)                    
                
                #have the "deviation", "number" and "week_last"：前一周末
                elif(matcher.group(2) and matcher.group(4) and matcher.group(3) != u'工作日' and matcher.group(5) == u''):
                    start_dev, end_dev = self.week_period_dev(matcher.group(1), matcher.group(4), matcher.group(2), basetime)   
                    
                #have the '工作日'
                elif(not matcher.group(2) and matcher.group(3) == u'工作日' and matcher.group(5) == u''):
                    start_dev, end_dev = self.week_period_dev(matcher.group(1), matcher.group(3), '', basetime)
                
                #only have the "deviation"
                elif(not matcher.group(2) and not matcher.group(4) and matcher.group(5) == u''):
                    start_dev, end_dev = self.week_period_dev(matcher.group(1), u'', '', basetime)
                else:
                    print("-----------------------------")
                    print("no matcher")
                    error = True
            else:
                #match the u'.*?(星期|礼拜|周)([末|六日]{0,})'
                matcher = re.match(self.pattern_last, entity)
                if(matcher):
                    start_dev, end_dev = self.week_period_dev(u'本', matcher.group(2), '', basetime)
                else:
                    #match the u'45个工作日内'
                    matcher_num_workday = re.match(self.pattern_num_workday, entity)
                    if(matcher_num_workday):    
                        start_dev, end_dev = self.workday_dev(matcher_num_workday.group(1), matcher_num_workday.group(2), matcher_num_workday.group(3))
                    else:
                        #match the u'工作日'
                        matcher = re.match(self.pattern_workday, entity)
                        if(matcher):
                            start_dev, end_dev = self.week_period_dev(u'本', matcher.group(1), '', basetime)
                        
                        else:
                            print("-----------------------------")
                            print("no matcher")
                            error = True
        
        return start_dev, end_dev, error
        

if __name__ == '__main__':
    week = Week()
#    entity = u'全周'
    entity = u'最近一周' #双休日
    basetime = u'2019-02-14 12:00:00'
    start_dev, end_dev, error = week.week_recognise(entity, basetime)
