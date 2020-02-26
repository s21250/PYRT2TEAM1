import pandas as pd
import os

import pandas as pd
import os

class acorn:
    acorn_file='acorn_details.csv'
    df=pd.read_csv(acorn_file, encoding='koi8_r')
    df.columns=['MAIN_CATEGORIES', 'CATEGORIES', 'REFERENCE', 'ACORN-A', 'ACORN-B',
                         'ACORN-C', 'ACORN-D', 'ACORN-E', 'ACORN-F', 'ACORN-G', 'ACORN-H',
                         'ACORN-I', 'ACORN-J', 'ACORN-K', 'ACORN-L', 'ACORN-M', 'ACORN-N',
                         'ACORN-O', 'ACORN-P', 'ACORN-Q']
    maincategories=df.MAIN_CATEGORIES.unique()
    #categories={}
    def __init__(self):
        __class__.categories={m:list(__class__.df[__class__.df.MAIN_CATEGORIES==m].CATEGORIES.unique()) for m in __class__.maincategories}