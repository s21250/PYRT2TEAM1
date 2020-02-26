#!/usr/bin/env python
# coding: utf-8

# In[4]:


#DEFINITIONS
from game import acorn as ac
from building import building as bd
import weather as w

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cufflinks as cf
import seaborn as sns

import ipywidgets as widgets
from ipywidgets import interact, HBox, Label, VBox, Layout, AppLayout
import plotly.graph_objects as go
#import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import iplot ,download_plotlyjs, init_notebook_mode, plot
get_ipython().run_line_magic('matplotlib', 'inline')
init_notebook_mode(connected = True)
cf.go_offline()

#Настраиваем вывод через display() датасетов
pd.options.display.show_dimensions=False
pd.options.display.min_rows=30
pd.options.display.max_rows=30
#STYLES
#Несколько layout для простоты настройки интерфейса
layout300=Layout(width='300px')
layout250=Layout(width='250px')
layout200=Layout(width='200px')

blocks=widgets.Dropdown(options=bd.directory_listing(),layout=layout200, continuous_update=False)
blocks_df=bd(blocks.options[0]).df
acorn_df = pd.read_csv('acorn_stat.csv') #Главный файл для работы. Требуется подготовка через 'acorn-prepare.py' или книгу 'acorn'
acorn_time_interval=set()
[acorn_time_interval.add((x.year,x.month)) for x in pd.to_datetime(acorn_df.day.unique())]
acorn_time_interval=sorted(list(acorn_time_interval))
acorn_time_interval=[pd.to_datetime(str(x)+'-'+str(y)) for (x,y) in acorn_time_interval]
#DEFINITIONS-END


# In[5]:


#ACORN-TAB
#TAB2
acorn_cats=list(acorn_df.Acorn.unique())
acorn_cats.insert(0, 'ALL')#
#TAB2-FUNC
def draw_acorn(change): #Обработчик виджетов для графика
    global acorn_df
    global tdfa
    tdf=acorn_df[(acorn_df.Acorn.isin(list(acorn_grp_selector.value)))
                 & (pd.to_datetime(acorn_df.day)>=date_range_min)
                 & (pd.to_datetime(acorn_df.day)<=date_range_max)
                ].copy()
    tdf3=tdf.groupby(['day','file', 'Acorn',stat_selector.value]).sum()
    tdf3=tdf3.reset_index()
    tdf4=[tdf3[tdf3.Acorn==x].groupby('day').sum() for x in tdf3.Acorn.unique()]
    tdf4=[x[[stat_selector.value]].reset_index() for x in tdf4]
    global traces
    traces=[go.Scatter(x=tdf4[i]['day'], y=tdf4[i][stat_selector.value]
                       , name=tdf.Acorn.unique()[i])
            for i in range(len(tdf.Acorn.unique()))]
    if 'ALL' in list(acorn_grp_selector.value):
        tdfa=acorn_df[(pd.to_datetime(acorn_df.day)>=date_range_min)
                 & (pd.to_datetime(acorn_df.day)<=date_range_max)
                ].copy()
        tdfa=tdfa.groupby('day').sum()
        tdfa=tdfa[[stat_selector.value]].reset_index()
        tracea=go.Scatter(x=tdfa['day'], y=tdfa[stat_selector.value], name='ALL')
        traces.append(tracea)
    global acorn_graph
    with acorn_graph.batch_update():
        acorn_graph.data=()
        for i in range(len(traces)):
            acorn_graph.add_trace(traces[i])
def acorn_update_cats(*args):
    global categories
    acorn_categories.options=ac.categories[acorn_maincategories.value]
def acorn_info(*arg):
    a=[blocks_df.Acorn.unique()][0]
    acorn_sel_cats=[x for x in a if x in acorn.df.columns]
    tbl_header=HBox([Label('REFERENCE', layout=th_layout)]+[Label(x, layout=th_layout) for x in acorn_sel_cats],layout=tr_layout)

    tdf=acorn.df[['MAIN_CATEGORIES','CATEGORIES','REFERENCE']+acorn_sel_cats][acorn.df['CATEGORIES']==acorn_categories.value]

    tbl_rows=[tbl_header]
    for i in range(len(tdf)):
        s=tdf.iloc[i]
        tbl_rows.append(HBox([Label((s['REFERENCE']+':'),layout=td_layout)]+[Label(str(s[x]),layout=td_layout) for x in acorn_sel_cats], layout=tr_layout))
    global acorn_stat_info
    acorn_stat_info=VBox(tbl_rows, layout=tbl_layout, style={'align':'center'})
    acorn_stat_info.layout=tbl_layout
    app.center=acorn_stat_info



#TAB2-WIDGETS
#NEST1
acorn_grp_selector=widgets.SelectMultiple(
    options=acorn_cats,
    rows=10,
    description='Какие категории сравниваем',
    disabled=False,
    style = {'description_width': 'initial'}
)
stat_selector=widgets.Select(
    options=['energy_median', 'energy_mean', 'energy_max', 'energy_std', 'energy_sum', 'energy_min'],
    value='energy_mean',
    rows=6,
    description='По какому показателю',
    disabled=False,
    style = {'description_width': 'initial'}
)
acorn_hbox=HBox([acorn_grp_selector,stat_selector], layout=Layout(min_width='100%'
                                                                  ,flex_flow='row'
                                                                  ,justify_content='center'))
acorn_graph = go.FigureWidget(
                    layout=go.Layout(
                        title=dict(
                            text='Сравнение категорий ACORN'
                        ),
                        barmode='overlay'
                    ))
acorn_graph_box=VBox([acorn_graph])
tab2=tab2_acorn=VBox([acorn_hbox])
#NEST2
acorn=ac()
tbl_layout=Layout(display='table', border='1px solid black', min_width='33%')
tr_layout=Layout(display='table-row',
                 align_items='flex-start',
                 align_content='flex-start',
                 justify_content='center',
                 border='1px solid green',
                 padding='2px'
              )
th_layout=Layout(display='table-cell'
                ,align_content='flex-end'
               , padding='2px 2px 2px 10px'
                 ,font_weight='bold'
                , border='1px solid black', min_width='33%')

td_layout=Layout(display='table-cell'
                ,align_content='flex-end'
               , padding='2px'
                , border='1px solid black', min_width='33%')
acorn_maincategories=widgets.Dropdown(options=acorn.maincategories, value=acorn.maincategories[0])
acorn_categories=widgets.Dropdown(options=acorn.categories[acorn_maincategories.value])
tab2_acorn_info=HBox([acorn_maincategories,acorn_categories])
acorn_stat_info=VBox(layout=Layout(display='none'))


tab2_block=widgets.Accordion(children=[tab2_acorn,tab2_acorn_info], layout=Layout(max_height='550px'))
tab2_block.set_title(0, 'Сравнение потребления энергии')
tab2_block.set_title(1, 'Справочная информация')

#TAB2-EVENTS
acorn_grp_selector.observe(draw_acorn,'value')
stat_selector.observe(draw_acorn,'value')
acorn_maincategories.observe(acorn_update_cats, 'value')
acorn_maincategories.observe(acorn_info, 'value')
acorn_categories.observe(acorn_info, 'value')
#TAB2-END


# In[6]:


#TIME
date_range_min=pd.to_datetime(acorn_df.day.min())
date_range_max=pd.to_datetime(acorn_df.day.max())
#TIME-FUNC
def date_range_upd(change):
    global date_range_min
    if date_range.value[0]==0:
        date_range_min=pd.to_datetime(acorn_df.day.min())
    else:
        date_range_min=acorn_time_interval[date_range.value[0]]
    global date_range_max
    if date_range.value[1]==len(acorn_time_interval):
        date_range_max=pd.to_datetime(acorn_df.day.max())
    else:
        date_range_max=acorn_time_interval[date_range.value[1]]-pd.DateOffset(days=1)+pd.DateOffset(months=1)
    date_range_lbl.value=date_range_min.strftime('%Y-%m-%d')+' - '+date_range_max.strftime('%Y-%m-%d')
#TIME-WIDGETS
date_range=widgets.IntRangeSlider(
    value=[0, len(acorn_time_interval)-1],
    min=0,
    max=len(acorn_time_interval)-1,
    step=1,
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=False,
    readout_format='d',
    layout=layout250

)
date_range_lbl=Label(date_range_min.strftime('%Y-%m-%d')+' - '+date_range_max.strftime('%Y-%m-%d'), style={'text-align':'right'})
date_range_box=VBox([date_range, date_range_lbl], layout=Layout(flex_flow='column'))
#TIME-EVENTS
date_range.observe(date_range_upd,'value')
date_range.observe(draw_acorn,'value')
#TIME-END


# In[7]:


#BLOCKS
#TAB1
blocks_df_interval=blocks_df
#TAB1-WIDGETS
#TAB1-WIDGETS-MAIN
tab1_main=VBox([Label('Выберите дом:')
                  ,blocks
                  ,Label('Выберите время:')
                 ,date_range_box]
                 ,layout=Layout(display='block',
                                border='solid 1px lightgrey',
                                width='400px',
                                max_height='400px'))
###
tab1_slider=widgets.IntSlider(
    value=0,
    min=0,
    max=len(blocks_df)-20,
    step=20,
    continuous_update=False,
    description='Просмотр строки:',
    orientation='horizontal',
    readout=True,
    readout_format='d',
    style={'description_width': 'initial'}
)
    #Вывод датасетов. Сначала бегунок - пробегает по строкам 
tab1_output=widgets.Output(layout=Layout(align_items='flex-start', align_content='flex-start'))

with tab1_output:
    @interact
    def asd(x=tab1_slider):
        return display(blocks_df_interval.loc[0+x:20+x])
    
tab1_lbl_full_strs=widgets.Label()
tab1_lbl_full_energy=widgets.Label()
tab1_lbl_acorn_cats=widgets.Label()
tab1_lbl_lclids=widgets.Label()
tab1_lbl_int_str=widgets.Label()
tab1_lbl_int_energy=widgets.Label()

tab1_lbl_info1=Label('Количество строк всего:')
tab1_lbl_info2=Label('Потрачено энергии:')
tab1_lbl_info3=Label('Категории ACORN:')
tab1_lbl_info4=Label('Количество датчиков:')
tab1_lbl_info5=Label('Количество строк за период:')   
tab1_lbl_info6=Label('Потрачено энергии за период:')

tab1_df_exam=VBox([tab1_output])
tab1_total_df=VBox([tab1_lbl_info1, tab1_lbl_full_strs
                     ,tab1_lbl_info2, tab1_lbl_full_energy
                     ,tab1_lbl_info3, tab1_lbl_acorn_cats
                     ,tab1_lbl_info4,  tab1_lbl_lclids]
                    ,layout=Layout(width='50%', border='solid 1px grey', padding='5px', margin='0 2px 0 0'))
tab1_interval_df=VBox([tab1_lbl_info5, tab1_lbl_int_str
                        ,tab1_lbl_info6, tab1_lbl_int_energy]
                       ,layout=Layout(width='50%', border='solid 1px grey', padding='5px', margin='0 0 0 2px'))
#tab1_block=HBox([tab1_main,tab1_total_df, tab1_interval_df])
tab1_block=HBox([tab1_total_df, tab1_interval_df])
 

#TAB1-FUNC
#def update_bld(*args):
def blocks_update(*arg):
    global blocks_df
    blocks_df=bd(blocks.value).df
    global blocks_df_interval
    blocks_df_interval=df_interval(blocks_df)
    tab1_output.clear_output()
    global tab1_slider
    tab1_slider.max=len(blocks_df_interval)
    tab1_slider.value=0
    update_tab1_whole(blocks_df)
    update_tab1_part(blocks_df_interval)
    #tab_switch()
    with tab1_output:
        @interact
        def asd(x=tab1_slider):
            return display(blocks_df_interval.loc[0+x:20+x])
def df_interval(df):
    return blocks_df[(pd.to_datetime(blocks_df.day)>=date_range_min) & (pd.to_datetime(blocks_df.day)<=date_range_max)]
def update_tab1_whole(df):
    tab1_lbl_full_strs.value=str(len(df))
    tab1_lbl_full_energy.value=str(round(df.energy_sum.sum(),3))
    tab1_lbl_acorn_cats.value=str([df.Acorn.unique()][0]).strip('[]')
    #acorn_multi1.value=list(df.Acorn.unique())
    tab1_lbl_lclids.value=str(len(df.LCLid.unique()))
def update_tab1_part(df_time):
    tab1_lbl_int_str.value=str(len(df_time))
    tab1_lbl_int_energy.value=str(round(df_time.energy_sum.sum(),3))
update_tab1_whole(blocks_df)
update_tab1_part(df_interval(blocks_df))

#TAB1-EVENTS
blocks.observe(blocks_update, 'value')
date_range.observe(blocks_update,'value')
#TAB1-END


# In[8]:


#WEATHER
#TAB3
weather_df=w.wdf_eng
temperature_cats=['apparentTemperatureHigh', 'apparentTemperatureLow',
                  'apparentTemperatureMax', 'temperatureHigh','apparentTemperatureMin',
                  'temperatureLow', 'temperatureMax', 'temperatureMin']
temperature_df=w.wdf[['day', 'weekday', 'week', 'year', 'apparentTemperatureHigh'
                      , 'apparentTemperatureLow', 'apparentTemperatureMax', 'apparentTemperatureMin'
                      , 'temperatureHigh','temperatureLow', 'temperatureMax', 'temperatureMin']]
#TAB3-FUNC
#Обработчик виджетов для графика
temper_graph = go.FigureWidget()
def draw_temperature(change): #Обработчик виджетов для графика
    global temperature_df
    global tdfa
    tc=list(temperature_selector.value)
    tc.append('day')
    tdf=temperature_df[tc][(pd.to_datetime(temperature_df.day)>=date_range_min)
                 & (pd.to_datetime(temperature_df.day)<=date_range_max)
                ].copy()
    tdf4=[tdf[['day',x]] for x in temperature_selector.value]
    tdf4=[x.reset_index() for x in tdf4]
    global traces
    traces=[go.Scatter(x=tdf4[i]['day'], y=tdf4[i][temperature_selector.value[i]]
                       , name=temperature_selector.value[i])
            for i in range(len(temperature_selector.value))]
    global temp_graph
    with temp_graph.batch_update():
        temp_graph.data=()
        for i in range(len(traces)):
            temp_graph.add_trace(traces[i])
            
def draw_temperature1(change): 
    global date_range_min
    global date_min
    global date_max
    if date_range_min.day!=1:
        date_min=(pd.to_datetime(date_range_min)+pd.DateOffset(months=1)).replace(day=1)
        date_max=(date_min+pd.DateOffset(months=1)).replace(day=1)-pd.DateOffset(days=1)
    else:
        date_min=pd.to_datetime(date_range_min)
        date_max=(date_min+pd.DateOffset(months=1))-pd.DateOffset(days=1)
    global ggg
    ggg=date_max
    date_min=date_min-pd.DateOffset(days=int(date_min.weekday()))
    date_max=date_max+pd.DateOffset(days=6-int(date_max.weekday()))

    #Формируем данные
    global map_df
    map_df=wdf[(pd.to_datetime(wdf.day)>=date_min) &(pd.to_datetime(wdf.day)<=date_max)]
    map_df.day=pd.to_datetime(map_df.day)
    #Делаем заглушку, если данных за какой-то из дней нет
    map_df=map_df.merge(pd.DataFrame({ 'day': pd.date_range(date_min, date_max)}), how='right')
    map_df.temperatureHigh=map_df.temperatureHigh.fillna(0)

    #Определяем данные для тепловой карты
    map_labels=np.flip(np.asarray(map_df['day'].dt.strftime('%Y-%m-%d')).reshape(int(len(map_df)/7), 7), 0) #7, int(len(df_range)/7)
    map_data=np.flip(np.asarray(map_df['temperatureMax']).reshape(int(len(map_df)/7), 7,), 0)
    labels=(np.asarray(["{0}<br>{1:.2f}".format(x,y) for x,y in zip(map_labels.flatten(),map_data.flatten())])).reshape(int(len(map_df)/7), 7)
    z=map_data

    x = [1,2,3,4,5,6,7]
    if len(map_df)==42:
        y = [1,2,3,4,5,6]
    else:
        y = [1,2,3,4,5]
    z=map_data

    annotations = go.Annotations()
    for n, row in enumerate(z):
        for m, val in enumerate(row):
            annotations.append(go.Annotation(text=str(labels[n][m])
                                             , x=x[m]
                                             , y=y[n]
                                             ,xref='x'
                                             , yref='y', showarrow=False))
    trace = go.Heatmap(x=x, y=y, z=z,colorscale='Plasma', showscale=True,hoverinfo='z')
    global temper_graph
    with temper_graph.batch_update():
        temper_graph.data=()
        temper_graph.add_trace(trace)
        temper_graph['layout'].update(
            title="Карта температуры за "+date_min.strftime('%Y-%m-%d')+' - '+date_max.strftime('%Y-%m-%d'),
            annotations=annotations,
            xaxis=go.XAxis(ticks='', side='top',showticklabels=False),
            yaxis=go.YAxis(ticks='', ticksuffix='  ',showticklabels=False),  # ticksuffix is a workaround to add a bit of padding
            autosize=True
        )
def draw_pairplot(change): #Обработчик виджетов для графика
    global blocks_df
    global weather_pairplot
    global weather_smulti
    df=pd.merge(df_interval(blocks_df)[['day','energy_sum']],df_range, on='day')
    index_vals = df['energy_sum'].astype('category').cat.codes
    
    
    
    trace=go.Splom(
        dimensions=[dict(label=weather_smulti.value[i],
                         values=df[weather_smulti.value[i]])
                    for i in range(len(weather_smulti.value))],
        text=df['energy_sum'],
        marker=dict(color=index_vals,
                    showscale=False,
                    line_color='white', line_width=0.5)
    )
    with weather_pairplot.batch_update():
        weather_pairplot.data=()
        weather_pairplot.add_trace(trace)
        weather_pairplot['layout'].update(
            title='Диаграмма рассеивания за '+str(date_range_min)+' - '+str(date_range_max),
            dragmode='select',
            width=1200,
            height=1200,
        hovermode='closest',
        )
#TAB3-WIDGETS
temperature_selector=widgets.SelectMultiple(
    options=temperature_cats,
    rows=5,
    description='Какие температуры смотрим:',
    disabled=False,
    style = {'description_width': 'initial'},
    layout=Layout(width='50%')
)
temp_graph = go.FigureWidget(
                    layout=go.Layout(
                        title=dict(
                            text='Динамика температуры'
                        ),
                        barmode='overlay'
                    ))
temp_graph_box=VBox([temp_graph])

#TAB3-EVENTS
temperature_selector.observe(draw_temperature, 'value')
date_range.observe(draw_temperature, 'value')
date_range.observe(draw_temperature1, 'value')


# In[10]:



def update_time_info(*args):
   weather_pie.clear_output()
   global piechart
   piechart=date2piechart(date_range_min)
   with weather_pie:
       display(piechart.get_figure())
   piechart.remove() 

#WEATHER-2
time_lbl=Label()
wdf=w.wdf
weather_range=w.dt2range(date_range_min)
df_range=w.df_range(date_range_min,wdf)
#awidgets.link((time_selector, 'value'), (time_lbl, 'value'))

#TAB3-2-FUNCS
#HEATMAP
def date2piechart(dt):
   global df_range
   df_range=w.df_range(dt,wdf)
   df_range1=w.df2rus(df_range)
   df_pie=pd.pivot_table(df_range1[['day', 'Направление ветра']], values='day', index=['Направление ветра'], aggfunc='count')
   df_set=df_pie.plot.pie(y='day', subplots=False, figsize=(5, 5))
   df_set.set_ylabel('')
   df_set.set_xlabel('')
   df_set.set_title('Направление ветра')
   return df_set
#piechart=date2piechart(time_selector.value)
def date2heatmap(dt):
   global df_range
   df_range=w.df_range(dt,wdf)
   map_labels=np.asarray(df_range['day']).reshape(7, int(len(df_range)/7))
   map_data=np.asarray(df_range['temperatureMax']).reshape(7, int(len(df_range)/7))
   labels=(np.asarray(["{0}\n{1:.2f}".format(x,y) for x,y in zip(map_labels.flatten(),map_data.flatten())])).reshape(7, int(len(df_range)/7))
   fix,ax=plt.subplots(figsize=(8,4))
   title='Температура за период'
   plt.title(title,fontsize=19)
   ttl=ax.title
   ttl.set_position([0.5,1.05])
   #ax.set_xticks([])
   #ax.set_yticks([])
   ax.axis('off')
   return sns.heatmap(map_data,annot=labels,fmt="",cmap='YlOrRd', linewidths=0.3, ax=ax, cbar=False)
def build_pp(*arg):
   weather_pairplot.clear_output()
   dfw=df_interval(blocks_df)[['day','energy_sum']]
   df1=pd.merge(dfw,df_range, on='day')
   pp = sns.pairplot(data=df1,
                     x_vars=['energy_sum'],
                     y_vars=list(weather_smulti.value),
                     palette="husl")
   pp.fig.set_size_inches(12,len(weather_smulti.value)*10)
   with weather_pairplot:
       display(pp.fig)
   app.center=weather_pairplot

#TAB3-2-WIDGETS


weather_pie=widgets.Output()
weather_pairplot= go.FigureWidget()
weather_pair_box=HBox([weather_pairplot])
#with weather_pie:
#    piechart=date2piechart(date_range_min)
#    display(piechart.get_figure())

#Основные сведения о погоде
weather_acc1=HBox([temper_graph]
                 ,layout=Layout(width='100%', border='solid 1px grey', padding='5px', margin='0 2px 0 0'))

weather_options=['temperatureMax', 'temperatureMaxTime',
      'windBearing', 'icon', 'dewPoint', 'temperatureMinTime', 'cloudCover',
      'windSpeed', 'pressure', 'apparentTemperatureMinTime',
      'apparentTemperatureHigh', 'precipType', 'visibility', 'humidity',
      'apparentTemperatureHighTime', 'apparentTemperatureLow',
      'apparentTemperatureMax', 'uvIndex', 'time', 'sunsetTime',
      'temperatureLow', 'temperatureMin', 'temperatureHigh', 'sunriseTime',
      'temperatureHighTime', 'uvIndexTime', 'summary', 'temperatureLowTime',
      'apparentTemperatureMin', 'apparentTemperatureMaxTime',
      'apparentTemperatureLowTime', 'moonPhase']
weather_values=['temperatureMax','windBearing', 'icon', 'dewPoint', 'cloudCover','windSpeed', 'pressure', 'visibility', 'humidity',
                         'temperatureLow', 'temperatureMin', 'temperatureHigh', 'summary']
#Диаграмма рассеивания
weather_smulti=widgets.SelectMultiple(
   options=weather_options,
   value=weather_values,
   rows=10,
   description='Выберите поля',
   disabled=False,
   style = {'description_width': 'initial'}
)
weather_button=widgets.Button(
   description='Построить',
   disabled=False,
   button_style='', # 'success', 'info', 'warning', 'danger' or ''
   tooltip='Построить диаграмму',
   icon='check'
)

#Диаграмма рассеивания
weather_vbox=VBox([weather_smulti,weather_button], layout=Layout(align_items='center',max_height='550px', height='300px'))
weather_hbox=HBox([weather_vbox], layout=Layout(align_items='flex-start',max_height='550px', height='300px'))

tab3_block=widgets.Accordion(children=[weather_acc1, weather_hbox,temperature_selector], layout=Layout(max_height='550px'))
tab3_block.set_title(0, 'Основные сведения о погоде')
tab3_block.set_title(1, 'Диаграмма рассеивания')
tab3_block.set_title(2, 'Динамика температур')

#TAB3-EVENTS
date_range.observe(update_time_info, 'value')
weather_button.on_click(draw_pairplot)


# In[11]:


#TABS
#TABS
tab_nest = widgets.Tab(layout=Layout(width='100%', height='auto'),align_content='flex-start')
tab_nest.children = [tab1_block, tab2_block, tab3_block]
tab_nest.set_title(0, 'Информация о наборе данных')
tab_nest.set_title(1, 'Информация ACORN')
tab_nest.set_title(2, 'Погода')
def tab_switch(*arg):
    tab1_output.clear_output()
    with tab1_output:
        if tab_nest.selected_index==0:
            @interact
            def asd(x=df_slider):
                return display(blocks_df.loc[0+x:20+x])
        elif tab_nest.selected_index==1:
            print('Tab 1')
        else:
            print('Tab 2')
def tab_switch(*arg):
    if tab_nest.selected_index==0:
        app.center=tab1_df_exam
    elif tab_nest.selected_index==1:
        if tab2_block.selected_index==0:
            app.center=acorn_graph_box
        elif tab2_block.selected_index==1:
            app.center=acorn_stat_info
    elif tab_nest.selected_index==2:
        if tab3_block.selected_index==1:
            app.center=weather_pair_box
        elif tab3_block.selected_index==2:
            app.center=temp_graph_box
        else:
            app.center=VBox()
    else:
        app.center=VBox()
#tab_nest.observe(tab_switch, 'selected_index')
tab_nest.observe(draw_temperature1, 'selected_index')
tab_nest.observe(tab_switch, 'selected_index')
tab2_block.observe(tab_switch, 'selected_index')
tab3_block.observe(tab_switch, 'selected_index')

#TABS-END


# In[16]:


csstyles=widgets.HTML(
    value="<style>div.output_subarea {max-width: 100% !important;}#notebook-container .code_cell:first-child .out_prompt_overlay {display: none;}.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab { flex: 0 1 500px; }body {overflow: hidden;}.prompt, #notebook-container .code_cell:first-child .input {display: none !important;}#notebook-container {display: block !important;}#notebook-container .code_cell:first-child .output {display: block !important;min-width: 100%;height: auto !important;padding: 5px;justify-self: stretch;}#notebook-container, #notebook, #ipython-main-app {padding: 0px !important;margin: 0px !important;min-width: 100%;}</style>",
)


# In[17]:


#APP
header=HBox([tab1_main,tab_nest])
app = AppLayout(header=header, center=tab1_df_exam, footer=csstyles, layout=Layout(align_items='flex-start', align_content='flex-start')) 
app.layout.height='auto'
app.layout.max_width='100%'
app.layout.min_width='100%'
#APP-END




