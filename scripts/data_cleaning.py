import numpy as np

'''data cleaning'''
def clean_data_1(data):
    print ('cleanong data...')
    #drop duplicat data
    data=data.drop_duplicates()

    #handle missing value (remove row with citical data)
    data=data.dropna(subset=['column_of_interest'], how='any')

    #fill missing value in other column with defualt value
    data.fillna({'default_column':0}, inplace=true)

    print("data cleaning completed")
    return data
def clean_data_2(data):
    #fill the missing numerical value with column mean
    data.fillna(data.mean(numerical_only=true), inplace=true)

    #outlier detection(with IQR methode)
def treat_outliers_with_mean(column):
    Q1 = column.quantile(0.25),
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound  = Q3 + 1.5 * IQR

    #replace the outliire with mean
    cloumn_mean = column_mean()
    column = np.where((column < lower_bound) | (column > upper_bound), column_mean, column)
    return column

