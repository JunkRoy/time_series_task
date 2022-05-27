#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@Project : arima_task 
@File    : average_module.py
@Author  : JunkRoy
@Date    : 2022/5/27 17:52 
"""

import pandas as pd

filename = '../data/arima_data.xls'
# forrecastnum = 5
data = pd.read_excel(filename, index_col=u'日期')


