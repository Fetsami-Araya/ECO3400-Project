"""
@Date: March 20, 2021
@Authors: Fetsami Araya
@Description: This file will estimate LASSO, Ridge, Elastic Net, Gradient Boosting Tree, SVM, and neural network
machine learning models to nowcast Canadian GDP from 1960 through to 2020.

In addition, the predictive performance of these models will be assessed and compared to the predicitive accuracy
of a benchmark AR(1) model.
"""

# Imports
import numpy as np
import pandas as pd
from data_clean import createMasterData

def readData(dataset_type='og'):
    master_data, master_data_some_na, master_data_no_na = createMasterData()
    master_data_dict = {'og':master_data, 'some_na':master_data_some_na, 'no_na':master_data_no_na}
    return master_data_dict[dataset_type]

print(readData())

