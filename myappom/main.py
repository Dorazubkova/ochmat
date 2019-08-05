#!/usr/bin/env python
# coding: utf-8

# In[1]:


import bokeh
from bokeh.server.server import Server as server
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, show, output_notebook

import pandas as pd
import numpy as np
from bokeh.models import ColumnDataSource, HoverTool, LassoSelectTool, Label, Title, ZoomInTool, ZoomOutTool
from bokeh.layouts import gridplot
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import ColumnDataSource, Figure
from bokeh.models.widgets import PreText, Paragraph, Select, Dropdown, RadioButtonGroup, RangeSlider, Slider, CheckboxGroup,HTMLTemplateFormatter,TableColumn, RadioGroup
import bokeh.layouts as layout
from bokeh.models.widgets import Tabs, Panel
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from bokeh.tile_providers import CARTODBPOSITRON


# In[2]:


matrix_base = pd.read_csv(r'myappom/base.csv', encoding='cp1251', sep = ';')
matrix_2020 = pd.read_csv(r'myappom/2021.csv', encoding='cp1251', sep = ';')
matrix_2021 = pd.read_csv(r'myappom/2022.csv', encoding='cp1251', sep = ';')


# In[3]:


matrix_base = matrix_base.set_index('zid')
matrix_base = matrix_base.stack().reset_index()
matrix_base.columns = ['site_id_from','site_id_to','value']

matrix_2020 = matrix_2020.set_index('zid')
matrix_2020 = matrix_2020.stack().reset_index()
matrix_2020.columns = ['site_id_from','site_id_to','value']

matrix_2021 = matrix_2021.set_index('zid')
matrix_2021 = matrix_2021.stack().reset_index()
matrix_2021.columns = ['site_id_from','site_id_to','value']


# In[4]:


metro = pd.read_csv(r'myappom/metro.csv', encoding='cp1251', sep = ';')
taz = pd.read_csv(r'myappom/taz.csv', encoding='cp1251', sep = ';')


# In[5]:


matrix_base = pd.merge(matrix_base, taz, how='inner', left_on = ['site_id_from'], right_on = ['taz_id']).rename(
columns = {'X':'X_from','Y':'Y_from' })
matrix_base = pd.merge(matrix_base, metro, how='inner', left_on = ['site_id_to'], right_on = ['site_name'])
matrix_base = matrix_base[['site_id_from', 'site_id_to', 'value', 'X_from', 'Y_from', 'site_id','X','Y']].rename(
columns = {'X':'X_to','Y':'Y_to' })
matrix_base = matrix_base[matrix_base['value'] != 0]

matrix_base.head()


# In[6]:


matrix_2020 = pd.merge(matrix_2020, taz, how='inner', left_on = ['site_id_from'], right_on = ['taz_id']).rename(
columns = {'X':'X_from','Y':'Y_from' })
matrix_2020 = pd.merge(matrix_2020, metro, how='inner', left_on = ['site_id_to'], right_on = ['site_name'])
matrix_2020 = matrix_2020[['site_id_from', 'site_id_to', 'value', 'X_from', 'Y_from', 'site_id','X','Y']].rename(
columns = {'X':'X_to','Y':'Y_to' })
matrix_2020 = matrix_2020[matrix_2020['value'] != 0]

matrix_2020.head()


# In[15]:


matrix_2021 = pd.merge(matrix_2021, taz, how='inner', left_on = ['site_id_from'], right_on = ['taz_id']).rename(
columns = {'X':'X_from','Y':'Y_from' })
matrix_2021 = pd.merge(matrix_2021, metro, how='inner', left_on = ['site_id_to'], right_on = ['site_name'])
matrix_2021 = matrix_2021[['site_id_from', 'site_id_to', 'value', 'X_from', 'Y_from', 'site_id','X','Y']].rename(
columns = {'X':'X_to','Y':'Y_to' })
matrix_2021 = matrix_2021[matrix_2021['value'] != 0]

matrix_2021.head()


# In[59]:


hover_from1 = HoverTool(tooltips=[("taz", "@site_id_from")], names=["label_from"])
hover_to1 = HoverTool(tooltips=[("metro", "@site_id_to")], names=["label_to"])

hover_from2 = HoverTool(tooltips=[("taz", "@site_id_from")], names=["label_from2"])
hover_to2 = HoverTool(tooltips=[("metro", "@site_id_to")], names=["label_to2"])

lasso_from1 = LassoSelectTool(select_every_mousemove=False)
lasso_to1 = LassoSelectTool(select_every_mousemove=False)

lasso_from2 = LassoSelectTool(select_every_mousemove=False)
lasso_to2 = LassoSelectTool(select_every_mousemove=False)

toolList_from1 = [lasso_from1,  'reset',  'pan','wheel_zoom', hover_from1]
toolList_to1 = [lasso_to1,  'reset',  'pan', 'wheel_zoom', hover_to1]

toolList_from2 = [lasso_from2,  'reset',  'pan','wheel_zoom', hover_from2]
toolList_to2 = [lasso_to2,  'reset',  'pan', 'wheel_zoom', hover_to2]


# In[60]:


cds_from1 = dict(X_from=[], 
                Y_from=[],
                site_id_from=[])


cds_to1 = dict(X_to=[], 
            Y_to=[],
            site_id=[],
            site_name=[])

cds_from2 = dict(X_from=[], 
                Y_from=[],
                site_id_from=[])


cds_to2 = dict(X_to=[], 
            Y_to=[],
            site_id=[],
            site_name=[])

source_from1 = ColumnDataSource(data = cds_from1)
source_to1 = ColumnDataSource(data = cds_to1)

source_from2 = ColumnDataSource(data = cds_from2)
source_to2 = ColumnDataSource(data = cds_to2)


# In[61]:


#рисуем графики
p1 = figure(x_range=(3948598, 4354485), y_range=(7307581, 7725406), x_axis_type="mercator", y_axis_type="mercator", tools=toolList_from1)
p1.add_tile(CARTODBPOSITRON)

#слой сайтов from
r1 = p1.circle(x = 'X_from',
         y = 'Y_from',
         source=source_from1,
        fill_color='navy',
        size=10,
        fill_alpha = 1,
        name = "label_from",
        nonselection_fill_alpha=1,
        nonselection_fill_color='gray')


# In[62]:


p_to1 = figure(x_range=(3948598, 4354485), y_range=(7307581, 7725406), x_axis_type="mercator", y_axis_type="mercator", tools=toolList_to1)
p_to1.add_tile(CARTODBPOSITRON)

t1 = p_to1.circle(x = 'X_to', 
                y = 'Y_to', 
                fill_color='green', 
                fill_alpha = 0.6, 
                line_color='tan', 
                line_alpha = 0.8, 
                name = "label_to",
                size=6 , 
                source = source_to1,
                   nonselection_fill_alpha = 0.6, 
                nonselection_fill_color = 'papayawhip', 
                nonselection_line_color = None)

t_to1 = p_to1.circle(x = [], y = [], fill_color='orange', fill_alpha = 0.6, 
                    line_color= None, line_alpha = 0.8, size=[], nonselection_line_color = None,
                   nonselection_fill_color = 'papayawhip') 

l1 = p_to1.text(x = [], y = [], text_color='black', text = [], text_font_size='8pt',
                         text_font_style = 'bold')

ds1 = r1.data_source
tds1 = t1.data_source

tds_to1 = t_to1.data_source
lds1=l1.data_source


# In[63]:


#рисуем графики
p2 = figure(x_range=(3948598, 4354485), y_range=(7307581, 7725406), x_axis_type="mercator", y_axis_type="mercator", tools=toolList_to2)
p2.add_tile(CARTODBPOSITRON)

#слой сайтов to
r2 = p2.circle(x = 'X_to',
         y = 'Y_to',
         source=source_to2,
        fill_color='navy',
        size=10,
        fill_alpha = 1,
        name = "label_to2",
        nonselection_fill_alpha=1,
        nonselection_fill_color='gray')


# In[64]:


p_from2 = figure(x_range=(3948598, 4354485), y_range=(7307581, 7725406), x_axis_type="mercator", y_axis_type="mercator", tools=toolList_from2)
p_from2.add_tile(CARTODBPOSITRON)

t2 = p_from2.circle(x = 'X_from', 
                y = 'Y_from', 
                fill_color='green', 
                fill_alpha = 0.6, 
                line_color='tan', 
                line_alpha = 0.8, 
                name = "label_from2",
                size=6 , 
                source = source_from2,
                   nonselection_fill_alpha = 0.6, 
                nonselection_fill_color = 'papayawhip', 
                nonselection_line_color = None)

t_from2 = p_from2.circle(x = [], y = [], fill_color='orange', fill_alpha = 0.6, 
                    line_color= None, line_alpha = 0.8, size=[], nonselection_line_color = None,
                   nonselection_fill_color = 'papayawhip') 

l2 = p_from2.text(x = [], y = [], text_color='black', text = [], text_font_size='8pt',
                         text_font_style = 'bold')

ds2 = r2.data_source
tds2 = t2.data_source

tds_from2 = t_from2.data_source
lds2=l2.data_source


# In[78]:


radio_group1 = RadioGroup(
        labels=["matrix_base", "matrix_2020", "matrix_2021"]) #только 1 вариант

radio_group2 = RadioGroup(
        labels=["matrix_base", "matrix_2020", "matrix_2021"]) #только 1 вариант


# In[79]:


show(radio_group1)


# In[66]:


prev_matrix_from = ['matrix']
def previous_matrix_from(matrix):
    prev_matrix_from.append(str(matrix))
    return prev_matrix_from


prev_matrix_to = ['matrix']
def previous_matrix_to(matrix):
    prev_matrix_to.append(str(matrix))
    return prev_matrix_to


def clear():
    new_data_text1 = dict()
    new_data_text1['x'] = []
    new_data_text1['y'] = []
    new_data_text1['text'] = []

    new_data1 = dict()
    new_data1['x'] = []
    new_data1['y'] = []
    new_data1['size'] = []
    
    return new_data1, new_data_text1


def null_selection_from():
    source_from1.selected.update(indices=[]) 
    
def null_selection_to():
    source_to2.selected.update(indices=[])
    
index_from = [[-1]]
def previous_idx_from(idx):
    index_from.append(idx)
    return index_from
    
index_to = [[-1]]
def previous_idx_to(idx):
    index_to.append(idx)
    return index_to


# In[67]:


def update1(attrname, old, new):
    
    m = radio_group1.active
    print(m)
    
    previous_matrix_from(m)
    
    if prev_matrix_from[-1] != prev_matrix_from[-2]:
        new_data1, new_data_text1 = clear()  
        null_selection_from()
        
    if m == 0:
        tbl = matrix_base
    elif m == 1:
        tbl = matrix_2020
    elif m == 2:
        tbl = matrix_2021

    tbl_from = tbl.drop_duplicates(['X_from','Y_from'])    

    cds_upd1 = dict(     X_from=list(tbl_from['X_from'].values), 
                        Y_from=list(tbl_from['Y_from'].values), 
                        site_id_from=list(tbl_from['site_id_from'].values))
    tbl_to = tbl.drop_duplicates(['X_to','Y_to']) 
    
    cds_upd2 = dict(     X_to=list(tbl_to['X_to'].values), 
                        Y_to=list(tbl_to['Y_to'].values), 
                        site_id_to=list(tbl_to['site_id_to'].values))

    #1
    source_from_sl = ColumnDataSource(data = cds_upd1)
    source_from1.data = source_from_sl.data

    #2
    source_to_sl = ColumnDataSource(data = cds_upd2)
    source_to1.data = source_to_sl.data

#     Time_Title1.text = "Матрица: " + m


radio_group1.on_change('active', update1)


# In[68]:


def update2(attrname, old, new):
    
    m = radio_group2.active
    print(m)
    
    previous_matrix_to(m)
    
    if prev_matrix_to[-1] != prev_matrix_to[-2]:
        new_data1, new_data_text1 = clear()  
        null_selection_to()
        
    if m == 0:
        tbl = matrix_base
    elif m == 1:
        tbl = matrix_2020
    elif m == 2:
        tbl = matrix_2021

    tbl_to = tbl.drop_duplicates(['X_to','Y_to'])
    
    
    cds_upd1 = dict(     X_to=list(tbl_to['X_to'].values), 
                        Y_to=list(tbl_to['Y_to'].values), 
                        site_id_to=list(tbl_to['site_id_to'].values))
    
    tbl_from = tbl.drop_duplicates(['X_from','Y_from'])

    cds_upd2 = dict(     X_from=list(tbl_from['X_from'].values), 
                        Y_from=list(tbl_from['Y_from'].values), 
                        site_id_from=list(tbl_from['site_id_from'].values))
    
    

    #1
    source_to_sl = ColumnDataSource(data = cds_upd1)
    source_to2.data = source_to_sl.data

    #2
    source_from_sl = ColumnDataSource(data = cds_upd2)
    source_from2.data = source_from_sl.data

#     Time_Title1.text = "Матрица: " + m


radio_group2.on_change('active', update2)


# In[69]:


def callback1(attrname, old, new): 
    
    idx = source_from1.selected.indices
    print(idx)
    
    if not idx:
        previous_idx_from([])
    else:
        previous_idx_from(idx)
    
    print(index_from)
        
    if index_from[-1] == []:
        new_data1, new_data_text1 = clear()  
        null_selection_from()
        
    m = radio_group1.active
    print(m)
    
    if m == 0:
        tbl = matrix_base
    elif m == 1:
        tbl = matrix_2020
    elif m == 2:
        tbl = matrix_2021 

    #таблица с выбранными индексами 
    df = pd.DataFrame(data=ds1.data).iloc[idx]
    
    print(df)
    
    df1 = pd.merge(df, tbl, how = 'inner', left_on = ['site_id_from'], right_on = ['site_id_from'])
    
    print(df1)
    
    df1['value_sum'] = df1.groupby(['X_to','Y_to'])['value'].transform(sum)
    df1 = df1.drop_duplicates(['X_to','Y_to'])
    
    print(df1) 
    
    new_data1 = dict()

    new_data1['x'] = list(df1['X_to'])
    new_data1['y'] = list(df1['Y_to'])
    new_data1['size'] = list(df1['value_sum']/10)
    tds_to1.data = new_data1  
    
    new_data_text1 = dict()
    new_data_text1['x'] = list(df1['X_to'])
    new_data_text1['y'] = list(df1['Y_to'])
    new_data_text1['text_0'] = list(round(df1['value_sum']))
    new_data_text1['text'] = [x if x != 0 else None for x in new_data_text1['text_0']]
    lds1.data = new_data_text1
          
source_from1.selected.on_change('indices', callback1)


# In[71]:


def callback2(attrname, old, new): 
    
    idx = source_to2.selected.indices
    print(idx)
    
    if not idx:
        previous_idx_to([])
    else:
        previous_idx_to(idx)
    
    print(index_to)
        
    if index_to[-1] == []:
        new_data1, new_data_text1 = clear()  
        null_selection_to()
        
    m = radio_group2.active
    print(m)
    
    if m == 0:
        tbl = matrix_base
    elif m == 1:
        tbl = matrix_2020
    elif m == 2:
        tbl = matrix_2021 

    #таблица с выбранными индексами 
    df = pd.DataFrame(data=ds2.data).iloc[idx]
    
    print(df)
    
    df1 = pd.merge(df, tbl, how = 'inner', left_on = ['site_id_to'], right_on = ['site_id_to'])
    
    df1['value_sum'] = df1.groupby(['X_from','Y_from'])['value'].transform(sum)
    df1 = df1.drop_duplicates(['X_from','Y_from'])
    
    print(df1.columns) 
    
    new_data1 = dict()

    new_data1['x'] = list(df1['X_from'])
    new_data1['y'] = list(df1['Y_from'])
    new_data1['size'] = list(df1['value_sum']/10)
    tds_from2.data = new_data1  
    
    new_data_text1 = dict()
    new_data_text1['x'] = list(df1['X_from'])
    new_data_text1['y'] = list(df1['Y_from'])
    new_data_text1['text_0'] = list(round(df1['value_sum']))
    new_data_text1['text'] = [x if x != 0 else None for x in new_data_text1['text_0']]
    lds2.data = new_data_text1
          
source_to2.selected.on_change('indices', callback2)


# In[72]:


layout1 = layout.row(p1, p_to1)
layout2 = layout.row(p2, p_from2)
layout3 = layout.row(radio_group1)
layout4 = layout.row(radio_group2)

layout5 = layout.row(layout1, layout3)
layout6 = layout.row(layout2, layout4)

tab1 = Panel(child=layout5, title='Фильтр корреспонденций "ИЗ"')
tab2 = Panel(child=layout6, title='Фильтр корреспонденций "В"')

tabs = Tabs(tabs=[tab1, tab2])

doc = curdoc() #.add_root(tabs)
#doc.theme = theme
doc.add_root(tabs)


# In[ ]:




