#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
@Project : arima_task 
@File    : arima_main.py
@Author  : JunkRoy
@Date    : 2022/5/11 16:57 
"""
from statsmodels.tsa.arima_model import ARIMA


class ArimaModel:
    def __init__(self):
        pass

    def fit_model_with_p_q(self, data, p=0, q=0):
        model = None
        model_info = ""
        if not isinstance(p, int) or p < 0:
            return model, "parameter \"p\" is illegal"
        if not isinstance(q, int) or q < 0:
            return model, "parameter \"q\" is illegal"

        try:
            model = ARIMA(data, (p, 1, q)).fit()
            return model,

        except Exception as e:
            return model, e
