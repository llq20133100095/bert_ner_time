# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 17:03:54 2018

@author: leolqli
@function: process the "time_period" type
"""
from common.entity import entity
import pandas as pd
import arrow
from time_Info.week import Week
from time_Info.season import Season
from CommonParser import CommonParser
from time_slot_value import TimeSlotValue
import re


class TimePeriodProcess(entity):
    def __init__(self):
        #super the class "entity"
        super(TimePeriodProcess, self).__init__()
        self.basetime = None
        
    def set_entity_basetime(self, dataframe):
        """
        Function:
            set the value of class "entity" and basetime.
        Parameters:
            1.dataframe: pd.dataframe, each rows in csv_file
        """
        self.entity = dataframe[u'entity']
        self.type = dataframe[u'type']
        self.abs_rel = dataframe[u'abs_rel']
        self.is_refer = dataframe[u'is_refer']
        self.is_freq = dataframe[u'is_freq']
        self.basetime = dataframe[u'basetime']
        self.start = dataframe[u'start']
        self.end = dataframe[u'end']
         
    def week_each_entity(self, entity, basetime, com_par):
        """
        Function:
            1.process each entity in week type
        Parameters:
            1.entity:str
        Return:
            1.com_par: CommonParser
            2.error: boolean, True: is error
        """
        week = Week()
        self.basetime = basetime
        start_dev, end_dev, error = week.week_recognise(entity, self.basetime)
        #shift the date
        com_par.startDate = com_par.date.shift(days=start_dev) \
            .replace(hour=0, minute=0, second=0)
        com_par.endDate = com_par.date.shift(days=end_dev) \
            .replace(hour=23, minute=59, second=59)
        return com_par, error
    
    
    def week_parse(self, data):
        """
        Function:
            parsing the "week" time_period in normalization
        Parameters:
            1.data: pd.dataframe
        """
        self.set_entity_basetime(data)
        week = Week()
        if(pd.isnull(self.basetime)):
            com_par = CommonParser()
            self.basetime = com_par.date.format("YYYY-MM-DD HH:mm")
        else:
            com_par = CommonParser(self.basetime)
            
        start_dev, end_dev, error = week.week_recognise(self.entity, self.basetime)
        #shift the date
        com_par.startDate = com_par.date.shift(days=start_dev) \
            .replace(hour=0, minute=0, second=0).format("YYYY-MM-DD HH:mm")
        com_par.endDate = com_par.date.shift(days=end_dev) \
            .replace(hour=23, minute=59, second=59).format("YYYY-MM-DD HH:mm")
        return com_par, error

    def season_each_entity(self, entity, basetime, com_par):
        """
        Function:
            1.process each entity in season type
        Parameters:
            1.entity:str
        Return:
            1.com_par: CommonParser
            2.error: boolean, True: is error
        """
        season = Season()
        self.basetime = basetime
       
        start_year, end_year, start_month, end_month, basetime, error = season.season_recognise(entity, self.basetime)
        if(not error):
            matcher = re.match(u'([\\d]{4}).*?', basetime)
            com_par.date = arrow.get(basetime).replace(year=int(matcher.group(1)))
            
            #set the date in startDate and endDate
            com_par.startDate = com_par.date \
                .replace(year=start_year, month=start_month, day=1)
            utc = arrow.utcnow().to("local").replace(year=end_year, month=end_month, day=1)
            #get the last day in a month
            end_day = utc.ceil('month').day
            com_par.endDate = com_par.date \
                .replace(year=end_year, month=end_month, day=end_day)
            
        return com_par, error
        
    def season_parse(self, data):
        """
        Function:
            parsing the "season" time_period in normalization
        Parameters:
            1.data: pd.dataframe
        """
        self.set_entity_basetime(data)
        season = Season()

        if(pd.isnull(self.basetime)):
            com_par = CommonParser()
            self.basetime = com_par.date.format("YYYY-MM-DD")
        else:
            com_par = CommonParser(self.basetime)
            
        start_year, end_year, start_month, end_month, basetime, error = season.season_recognise(self.entity, self.basetime)
        if(not error):
            matcher = re.match(u'([\\d]{4}).*?', basetime)
            com_par.date = arrow.get(basetime).replace(year=int(matcher.group(1)))
            
            #set the date in startDate and endDate
            com_par.startDate = com_par.date \
                .replace(year=start_year, month=start_month, day=1).format("YYYY-MM-DD")
            utc = arrow.utcnow().to("local").replace(year=end_year, month=end_month, day=1)
            #get the last day in a month
            end_day = utc.ceil('month').day
            com_par.endDate = com_par.date \
                .replace(year=end_year, month=end_month, day=end_day).format("YYYY-MM-DD")
            
        return com_par, error
        
    def time_period_parse(self):
        """
        Function:
            parsing the "season" time_period in normalization
        Parameters:
            1.data: pd.dataframe
        """
        ent = entity()
        ent.entity = u'近半个月中'
        ent.type = "time_point"
        ent.abs_rel = "relative"
        ent.is_refer = False
        ent.is_freq = False
        time_slot_value = TimeSlotValue(ent, '2018-11-01 12:00:00')
        time = time_slot_value.parse()
        print("result:" + time.date)
        print('start:' + time.startDate)
        print('end:' + time.endDate)
        print str(time.timeUnit)
        print time.timeFormatInfo
        
        
if __name__ == '__main__':
    #instantiate the class "TimePeriodProcess"
    time_period_process = TimePeriodProcess()
    time_period_process.time_period_parse()

    
#    #process the week time_period
#    file_name = './test/week_test_data.csv'
#    file_dataframe = pd.read_csv(file_name, encoding = u'utf-8')
#    line_num = 0
#    dataframe_list = []
#    for idx, data in file_dataframe.iterrows():
#        each_line = []
#        com_par, error = time_period_process.week_parse(data)
#        
#        each_line.append(time_period_process.entity)
#        each_line.append(com_par.date.format("YYYY-MM-DD HH:mm"))
#        if com_par and not error:
#            print("-----------------------------")
#            print("entity:" + time_period_process.entity)
#            print("result:" + com_par.date.format("YYYY-MM-DD HH:mm"))
#            print('start:' + com_par.startDate)
#            print('end:' + com_par.endDate)
#            each_line.append(com_par.startDate)
#            each_line.append(com_par.endDate)
#        else:
#            each_line.append('')
#            each_line.append('')
#            
#        dataframe_list.append(each_line)
#    
#    save_data = pd.DataFrame(dataframe_list, columns=['entity', 'basetime', 'start', 'end'])
#    save_data.to_csv('./test/week_test_data_predict.csv', encoding='gb2312')
    
#    #process the season time_period
#    file_name = './test/season_test_data.csv'
#    file_dataframe = pd.read_csv(file_name, encoding = u'utf-8')
#    line_num = 0
#    dataframe_list = []
#    for idx, data in file_dataframe.iterrows():
#        each_line = []
#        com_par, error = time_period_process.season_parse(data)
#        
#        each_line.append(time_period_process.entity)
#        if(com_par and not error):
#            print("-----------------------------")
#            print("entity:" + time_period_process.entity)
#            print("result:" + com_par.date.format("YYYY-MM-DD HH:mm"))
#            print('start:' + com_par.startDate)
#            print('end:' + com_par.endDate)
#            each_line.append(com_par.date.format("YYYY-MM-DD HH:mm"))
#            each_line.append(com_par.startDate)
#            each_line.append(com_par.endDate)
#        
#        else:
#            each_line.append(com_par.date.format("YYYY-MM-DD HH:mm"))
#            each_line.append('')
#            each_line.append('')
#        
#        dataframe_list.append(each_line)
#    
#    save_data = pd.DataFrame(dataframe_list, columns=['entity', 'basetime', 'start', 'end'])
#    save_data.to_csv('./test/season_test_data_predict.csv', encoding='gb2312')

        