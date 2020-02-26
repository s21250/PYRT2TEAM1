import pandas as pd
import os

class building:
    is_daily=True
    info=pd.read_csv('informations_households.csv')
    buildings_list=list(info.file.unique())
    path_daily='daily_dataset/'
    path_halfhourly='halfhourly_dataset/'
    dailyfiles=[f for r,d,f in os.walk(path_daily)][0]
    halfhourlyfiles=[f for r,d,f in os.walk(path_halfhourly)][0]
    def __init__(self, file='block_0.csv'):
        self.file=file
        if '.csv' not in file:
            self.file+='.csv'
            self.name=file
        else:
            self.name=file[:-4]
        self.df=__class__.get_df(self)
    def get_df(self):
        if __class__.is_daily:
            df=pd.read_csv(__class__.path_daily+self.file)
        else:
            df=pd.read_csv(__class__.path_halfhourly+self.file)
        df=df.merge(__class__.info, on='LCLid', how='left')
        return df
    @staticmethod
    def directory_listing(is_daily=is_daily):
        if is_daily:
            return [f[0:-4] for f in __class__.dailyfiles]
        return [f[0:-4] for f in __class__.halfhourlyfiles]
    def change_ftype(self):
        if __class__.is_daily:
            __class__.is_daily=False
        else:
            __class__.is_daily=True