# ECO 3400 Project: Nowcasting

## Introduction
This code implements a variety of machine learning models with the goal of nowcasting Canadian GDP. 

## Download Instructions

Download the code as a ZIP file by clicking the green 'Clone or download' button and selecting 'Download ZIP'.


## File and folder description

* `data/`: This folder will contain all our data. Datasets were compiled from a variety of sources listed in the Data Sources section:
* `Results/`: Folder that contains the results of our analysis.
* `data_clean.py`: File
* `Master Datasets/`: Folder that contains the csv *M-ACTUAL.csv* which contains the full set of training data used to train and test our models.
* `nowcast_ml.py`: Main file that performs nowcasting.
* `dm_test.py`: File from [Johnathan Tsang (2017.)](google.com) that performs the Diebold-Mariano (1995) Test.

## Data Sources

|              Variable             | Frequency | Earliest Year | Year End |    Span    | Lag |
|:---------------------------------:|:---------:|:-------------:|:--------:|:----------:|:---:|
| Real GDP (dependent variable)     | Quarterly |          1961 | 2020     | 59 years   |     |
| Exports                           | Quarterly |          1961 | 2020     | 59 years   | 1Q  |
| Imports                           | Quarterly |          1961 | 2020     | 59 years   | 1Q  |
| Consumption                       | Quarterly |          1961 | 2020     | 59 years   | 1Q  |
| Gross Fixed Capital Formation     | Quarterly |          1961 | 2020     | 59 years   | 1Q  |
| Population                        | Quarterly |          1961 | 2020     | 59 years   | 1Q  |
| Household Saving rate             | Quarterly |          1961 | 2020     | 59 years   | 1Q  |
| Household Disposable Income       | Quarterly |          1961 | 2020     | 59 years   | 1Q  |
| Wages and Salaries                | Quarterly |          1961 | 2020     | 59 years   | 1Q  |
| Consumer Price Index              | Monthly   |          1961 | 2021     | 60 years   | 2M  |
| Industrial Product Price Index    | Monthly   |          1961 | 2021     | 60 years   | 2M  |
| Unemployment Rate                 | Monthly   |          1976 | 2021     | 45 years   | 2M  |
| West Texas Intermediate (WTI)     | Monthly   |          1983 | 2021     | 38 years   | 2M  |
| Housing Starts                    | Monthly   |          1990 | 2021     | 31 years   | 2M  |
| Retail Sales                      | Monthly   |          1991 | 2020     | 29 years   | 3M  |
| Manufacturing Sales               | Monthly   |          1992 | 2020     | 28 years   | 3M  |
| USD/CAD Exchange Rate             | Daily     |          1971 | 2021     | 50 years   |     |
| S&P/TSX Composite                 | Daily     |          1979 | 2021     | 42 years   |     |
| Long Term Bond Yield (> 10 years) | Daily     |          1990 | 2021     | 31 years   |     |
| BOC Target Rate                   | Daily     |          1992 | 2021     | 29 years   |     |
| Median Age                        | Annually  |          1971 | 2020     | 49 years   | 1Y  |
|                        Span Used: | Daily     |          1992 | 2020     | ~ 28 years |     |
|                                   |           |               |          |            |     |

## Required software and versioning
TBD

