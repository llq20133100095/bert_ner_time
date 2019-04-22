# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 15:05:52 2019

@author: leolqli
@Function: process the "time_frequency" type. call the "frequency.py"
"""
from common.entity import entity
from time_Info.frequency import Frequency
from time_Info.approximation import Approximation
from CommonParser import CommonParser
import pandas as pd


class TimeFrequencyProcess(entity):
    def __init__(self):
        #super the class "entity"
        super(TimeFrequencyProcess, self).__init__()
        
    def set_entity(self, dataframe):
        """
        function:
            set the value of class "entity" and basetime.
        Parameters:
            1.dataframe: pd.dataframe, each rows in csv_file
        """
        self.entity = dataframe[u'entity']
        self.type = dataframe[u'type']
        self.abs_rel = dataframe[u'abs_rel']
        self.is_refer = dataframe[u'is_refer']
        self.is_freq = dataframe[u'is_freq']
        
    def frequency_parse(self, data):
        """
        function:
            parsing the time_frequency in normalization
        Parameters:
            1.data: pd.dataframe
        """
        self.set_entity(data)
        frequency = Frequency()
        com_par = CommonParser()
        
        time_fre_list, error = frequency.frequency_recognise(self.entity)
        com_par.timeLength = time_fre_list
        
        return com_par, error
    
if __name__ =='__main__':
    
    time_frequency_process = TimeFrequencyProcess()
    #check if the approximation u"前后|左右"
    approximation = Approximation()
    
    #process the week time_period
    file_name_frequency = './test/time_frequency_test_data.csv'
    file_dataframe = pd.read_csv(file_name_frequency, encoding='utf-8')
    
    dataframe_list = []
    for idx, data in file_dataframe.iterrows():
        each_line = []
        com_par, error = time_frequency_process.frequency_parse(data)
        com_par.is_approximation = approximation.approximation_recognise(time_frequency_process.entity)
        
        each_line.append(time_frequency_process.entity)
        if(com_par and not error):
            print("-----------------------------")
            print("entity:" + time_frequency_process.entity)
            print("result:" + str(com_par.timeLength))
            each_line.append(com_par.timeLength)
            
        else:
            each_line.append('')
        
        each_line.append(com_par.is_approximation)
        dataframe_list.append(each_line)
    
    save_data = pd.DataFrame(dataframe_list, columns=['entity', 'result', 'is_approximation'])
    save_data.to_csv('./test/time_frequency_test_data_predicte.csv', encoding='gb2312')
