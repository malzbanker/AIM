import pandas as pd

def read_csv(file_path):
    '''read data from csv file'''
    print("extracting data from csv...")
    try:
        data=pd.read_csv(file_path)
        return data
    except Exception as e:
        print (f"error during data extarction: {e}")
        return None