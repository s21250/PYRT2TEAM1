import pandas as pd
def df_interval(df, timeint):
    df=df[(pd.to_datetime(df.day)<=pd.to_datetime(timeint)-pd.DateOffset(days=1)+pd.DateOffset(months=1))]
    df=df[pd.to_datetime(timeint)<=pd.to_datetime(df.day)]
    return df
def eng2rus(row):
    if row.precipType=='rain':
        row.precipType='дождь'
    elif row.precipType=='snow':
        row.precipType='снег'
    else:
        row.precipType='-'
    if row.icon=='fog':
        row.icon='туман'
    elif row.icon=='partly-cloudy-day':
        row.icon='облачный день'
    elif row.icon== 'wind':
        row.icon='ветренно'
    elif row.icon=='partly-cloudy-night':
        row.icon='облачная ночь'
    elif row.icon=='clear-day':
        row.icon='ясно'
    elif row.icon=='cloudy':
        row.icon='облачно'
    else:
        row.icon='-'
    if 25<=row.windBearing<70:
        row.windBearing='С-В'
    elif 70<=row.windBearing<115:
        row.windBearing='В'
    elif 115<=row.windBearing<160:
        row.windBearing='Ю-В'
    elif 160<=row.windBearing<205:
        row.windBearing='Ю'
    elif 205<=row.windBearing<250:
        row.windBearing='Ю-З'
    elif 250<=row.windBearing<295:
        row.windBearing='З'
    elif 295<=row.windBearing<340:
        row.windBearing='С-З'
    else:
        row.windBearing='С'
    row.sunriseTime=pd.to_datetime(row.sunriseTime)
    row.sunsetTime=pd.to_datetime(row.sunsetTime)
    row.windSpeed=1.609*row.windSpeed
    row.cloudCover=row.cloudCover*100
    row.visibility=round(1.609*row.visibility,2)
    row.humidity=row.humidity*100
    return row

wdf = pd.read_csv('weather_daily_darksky.csv')
wdf['day']=pd.to_datetime(wdf.sunriseTime).dt.strftime('%Y-%m-%d')
wdf['weekday']=pd.to_datetime(wdf['day']).dt.weekday+1
wdf['week']=pd.to_datetime(wdf['day']).dt.week
wdf['year']=pd.to_datetime(wdf['day']).dt.year

wdf=wdf.sort_values(by=['day']).reset_index(drop=True)
wdf_eng=wdf[['day', 'weekday', 'week','year', 'sunriseTime', 'sunsetTime', 'icon', 'temperatureMax', 'temperatureMin',
           'windBearing', 'windSpeed', 'cloudCover', 'visibility', 'precipType',
           'pressure', 'humidity']].copy()
columns_rus=['day', 'weekday', 'week','year', 'Рассвет', 'Закат', 'Иконка', 'Макс. t°C', 'Мин.  t°C', 'Направление ветра', 'Скорость ветра, км/ч', 'Облачность, %', 'Видимость, км', 'Тип осадков'
             , 'Давление', 'Отн. влажность, %']
wdf_rus=wdf_eng.copy()
wdf_rus=wdf_rus.apply(lambda row: eng2rus(row), axis=1)
wdf_rus.columns=columns_rus

def dt2range(dt):
    dt=pd.to_datetime(dt)
    start_date=dt-pd.DateOffset(days=dt.day-1)
    #print(start_date)
    if start_date.weekday()>0:
        start_date=start_date-pd.DateOffset(days=start_date.weekday())
        #print(start_date)
    stop_date=dt+pd.DateOffset(months=1)-pd.DateOffset(days=dt.day-1)
    #print(stop_date)
    if stop_date.weekday()<6:
        stop_date=stop_date+pd.DateOffset(days=6-stop_date.weekday())
    #print(stop_date)
    return pd.date_range(start_date, stop_date)
def df_range(dt, df):
    dt_range=dt2range(dt)
    return df[(pd.to_datetime(df.day)>=dt_range[0]) & (pd.to_datetime(df.day)<=dt_range[-1])]
def df2rus(df):
    df=df[['day', 'weekday', 'week','year', 'sunriseTime', 'sunsetTime', 'icon', 'temperatureMax', 'temperatureMin',
           'windBearing', 'windSpeed', 'cloudCover', 'visibility', 'precipType',
           'pressure', 'humidity']]
    df=df.apply(lambda row: eng2rus(row), axis=1)
    df.columns=columns_rus
    return df