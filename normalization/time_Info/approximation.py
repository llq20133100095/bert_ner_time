# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:06:25 2019

@author: leolqli
@Function: process the "approximation" type. These samples like "前后|左右". 
It is a basic class.
"""
import re 

class Approximation:
    def __init__(self):
        self.pattern_appr = u'.*?(前后|左右).*?'
    
    def approximation_recognise(self, entity):
        """
        Function:
            recognise the "approximation" type in given pattern
        Parameters:
            1.entity: str,
        Return:
            1.is_approximation: boolean, "True" is the approximation time.
        """
        is_approximation = False

        matcher_appr = re.match(self.pattern_appr, entity) 
        
        if(matcher_appr):
            is_approximation = True
            
        return is_approximation
    
if __name__ == "__main__":
    approximation = Approximation()
    entity = u"7天左右"
    is_approximation = approximation.approximation_recognise(entity)