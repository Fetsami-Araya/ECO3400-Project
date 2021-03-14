"""
@Date: March 12, 2021
@Authors: Fetsami Araya, Sam Kirsh, Matthew Rayka
@Description: Creation of master dataframe containing all the economic data that are
going to be used to train the machine learning nowcasting models. Many of the series are at different
frequencies, so each are upsampled using forward fill to fit a daily frequency.
"""

# Imports
import numpy as np
import pandas as pd


def readCEER():
    """
    Create dataframe of canadian effective exchange rate data
    """
    file_name = './data/CEER.csv'
    ceer = pd.read_csv(file_name,skiprows=11)
    ceer['date'] = pd.to_datetime(ceer['date'])
    ceer = ceer.iloc[:,:2]
    ceer = ceer.set_index('date')
    ceer.columns = ['CEER']
    return ceer.resample('D').ffill()

def readCPI():
    """
    Create dataframe of Consumer Price Index data
    """
    file_name = './data/CPI.csv'
    cpi = pd.read_csv(file_name)
    cpi = cpi[cpi["Products and product groups"]=='All-items']
    cpi = cpi[['REF_DATE','VALUE']]
    cpi['REF_DATE'] = pd.to_datetime(cpi['REF_DATE'])
    cpi.columns = ['date','CPI']
    cpi = cpi.set_index('date')
    return cpi.resample('D').ffill()

def readExports():
    """
    Create dataframe of exports data
    """
    file_name = './data/exports.csv'
    exports = pd.read_csv(file_name)
    exports = exports[exports['Trade']=='Export']
    exports = exports[['REF_DATE','VALUE']]
    exports['REF_DATE'] = pd.to_datetime(exports['REF_DATE'])
    exports.columns = ['date','Exports']
    exports = exports.set_index('date')
    return exports.resample('D').ffill()

def readConsumption():
    """
    Create dataframe of final consumption data
    """
    file_name = './data/finalConsumption.csv'
    consumption = pd.read_csv(file_name)
    consumption = consumption[consumption['Estimates']=='Household final consumption expenditure [C]']
    consumption = consumption[['REF_DATE','VALUE']]
    consumption['REF_DATE'] = pd.to_datetime(consumption['REF_DATE'])
    consumption.columns = ['date','Consumption']
    consumption = consumption.set_index('date')
    return consumption.resample('D').ffill()

def readGSPTSE():
    """
    Create dataframe of S&P/TSX composite data
    """
    file_name = './data/GSPTSE.csv'
    GSPTSE = pd.read_csv(file_name)
    GSPTSE = GSPTSE[['Date','Adj Close']]
    GSPTSE['Date'] = pd.to_datetime(GSPTSE['Date'])
    GSPTSE.columns = ['date','GSPTSE']
    GSPTSE = GSPTSE.set_index('date')
    return GSPTSE.resample('D').ffill()

def readHousingStarts():
    """
    Create dataframe of housing starts data
    """
    file_name = './data/housingStarts.csv'
    starts = pd.read_csv(file_name)
    starts = starts[starts['Type of unit']=='Total units']
    starts = starts[['REF_DATE','VALUE']]
    starts['REF_DATE'] = pd.to_datetime(starts['REF_DATE'])
    starts.columns = ['date','Housing Starts']
    starts = starts.set_index('date')
    return starts.resample('D').ffill()

def readInitialJobless():
    """
    Create dataframe of initial jobless claims data
    """
    file_name = './data/initialJobless.csv'
    jobless = pd.read_csv(file_name)
    jobless = jobless[jobless['GEO']=='Canada']
    jobless = jobless[['REF_DATE','VALUE']]
    jobless['REF_DATE'] = pd.to_datetime(jobless['REF_DATE'])
    jobless.columns = ['date','Jobless']
    jobless = jobless.set_index('date')
    return jobless.resample('D').ffill()

def readIPPI():
    """
    Create dataframe of industrial product price index data
    """
    file_name = './data/IPPI.csv'
    IPPI = pd.read_csv(file_name)
    IPPI = IPPI[IPPI['North American Product Classification System (NAPCS)']=="Total, Industrial product price index (IPPI)"]
    IPPI = IPPI[['REF_DATE','VALUE']]
    IPPI['REF_DATE'] = pd.to_datetime(IPPI['REF_DATE'])
    IPPI.columns = ['date','IPPI']
    IPPI = IPPI.set_index('date')
    return IPPI.resample('D').ffill()

def readPolicyRates():
    """
    Create dataframe of Bank of Canada daily nominal interest rate data
    """
    file_name = './data/policyRates.csv'
    rates = pd.read_csv(file_name)
    rates.columns = ['date','Policy Rate']
    rates['date'] = pd.to_datetime(rates['date'])
    rates = rates.set_index('date')
    return rates.resample('D').ffill()

def readGDP():
    """
    Create dataframe of real GDP data
    """
    file_name = './data/realGDP.csv'
    GDP = pd.read_csv(file_name)
    GDP = GDP[GDP['Estimates']=="Final consumption expenditure"]
    GDP = GDP[['REF_DATE','VALUE']]
    GDP['REF_DATE'] = pd.to_datetime(GDP['REF_DATE'])
    GDP.columns = ['date','GDP']
    GDP = GDP.set_index('date')
    return GDP.resample('D').ffill()

def readRetailTrade():
    """
    Create dataframe of retail trade data
    """
    file_name = './data/retailTrade.csv'
    retail = pd.read_csv(file_name)
    retail = retail[retail['GEO']=='Canada']
    retail = retail[['REF_DATE','VALUE']]
    retail['REF_DATE'] = pd.to_datetime(retail['REF_DATE'])
    retail.columns = ['date','Retail']
    retail = retail.set_index('date')
    return retail.resample('D').ffill()

def readUnemployment():
    """
    Create dataframe of unemployment data
    """
    file_name = './data/unemployment.csv'
    unemployment = pd.read_csv(file_name)
    unemployment = unemployment[unemployment["Labour force characteristics"]=='Unemployment rate']
    unemployment = unemployment[['REF_DATE','VALUE']]
    unemployment['REF_DATE'] = pd.to_datetime(unemployment['REF_DATE'])
    unemployment.columns = ['date','Unemployment']
    unemployment = unemployment.set_index('date')
    return unemployment.resample('D').ffill()

def readWCS():
    """
    Create dataframe for Western Canadian Standard oil prices
    """
    file_name = './data/WCS.csv'
    all_oil = pd.read_csv(file_name)
    WTI = all_oil[all_oil['Type']=='WTI']
    WTI = WTI[['When','Alberta']]
    WTI.columns = ['date','WTI']
    WTI['date'] = pd.to_datetime(WTI['date'])
    WTI = WTI.set_index('date')
    WTI = WTI.resample('D').ffill()

    return WTI

def readCAN10Y():
    """
    Create dataframe of Canadian 10-Year government bond yields
    """
    file_name = './data/Can10Y.csv'
    can10y = pd.read_csv(file_name)
    can10y.columns = ['date','10Y Bond Yield']
    can10y['date'] = pd.to_datetime(can10y['date'])
    can10y = can10y.set_index('date')
    return can10y.resample('D').ffill()

def readManufacturing():
    file_name = './data/'
    manufacturing = pd.read_csv(file_name)
    return

def readMktIncome():
    file_name = './data/medianMarketIncome.csv'
    mktincome = pd.read_csv(file_name)
    mktincome = mktincome[(mktincome["Income concept"]=='Median market income')&(mktincome['Economic family type']=="Economic families and persons not in an economic family")&(mktincome['UOM']=="2018 constant dollars")]
    mktincome = mktincome[['REF_DATE','VALUE']]
    mktincome['REF_DATE'] = pd.to_datetime(mktincome['REF_DATE'],format='%Y')
    mktincome.columns = ['date','Median Market Income']
    mktincome = mktincome.set_index('date')
    return mktincome

def readPopulation():
    file_name = './data/population.csv'
    pop = pd.read_csv(file_name)
    return pop

def readGoogleTrends():
    file_name = './data/unemploymentsearchGT.csv'
    google_trends = pd.read_csv(file_name,skiprows = 2)
    google_trends.columns = ['date','Unemployment Searches GT']
    google_trends['date'] = pd.to_datetime(google_trends['date'])
    google_trends = google_trends.set_index('date')
    return google_trends

def createMasterData():
    """
    Create master data set of all series
    """
    print('READING IN RAW DATA','\n')
    can10y = readCAN10Y()
    ceer = readCEER()
    cpi = readCPI()
    exports = readExports()
    consumption = readConsumption()
    gsptse = readGSPTSE()
    housing = readHousingStarts()
    jobless = readInitialJobless()
    ippi = readIPPI()
    rates = readPolicyRates()
    gdp = readGDP()
    retail = readRetailTrade()
    unemployment = readUnemployment()
    wcs = readWCS()
    print('\n','BEGINNING MERGE','\n')
    all_series_no_gdp = [unemployment, cpi, exports, 
                consumption, gsptse, housing, 
                jobless, ippi, rates, can10y,
                retail, ceer, wcs]
    
    # Using repeated joins to maximize data retention.
    for df in all_series_no_gdp:
        gdp = gdp.join(df)
    
    master_data = gdp.copy()
    master_data_no_na = master_data.dropna(how='any')
    master_data_some_na = master_data.dropna(how='all')

    print('\n','MERGE COMPLETE')

    return master_data, master_data_some_na, master_data_no_na

master_data, master_data_some_na, master_data_no_na = createMasterData()

print('\n\n','\t\t\t\t\t\t\t\t RAW','\n\n',
    master_data,
 '\n\n','\t\t\t\t\t\t\t\t SOME NAs','\n\n',
 master_data_some_na, 
 '\n\n','\t\t\t\t\t\t\t\t NO NAs','\n\n',
 master_data_no_na)
