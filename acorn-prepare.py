from building import building as bd
import pandas as pd

t=bd.directory_listing()
result_df=bd(t[0]).df
result_df=result_df.groupby(['day','Acorn', 'file']).sum()
for i in range(1,len(t)):
    df=bd(t[i]).df
    df=df.groupby(['day','Acorn', 'file']).sum()
    #print(df.info(verbose = False, null_counts = False))
    result_df=result_df.append(df)
result_df.reset_index()
result_df.to_csv('acorn_stat.csv')
result_df.info(verbose = False, null_counts = False)
print('Well done!')