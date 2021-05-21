# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as io
from sklearn.cluster import KMeans
import seaborn as sb
import gunicorn


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
data_clean = pd.read_csv('data_clean.csv')
data_all = pd.read_csv('data_all.csv')
data_clust = pd.read_csv('data_clust.csv')
data_exp = pd.read_csv('data_exp.csv')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server
app.config['suppress_callback_exceptions'] = True


    

app.layout = html.Div([
    html.H1('Project 2 - Civil Building'),
    html.Img(src=app.get_url('IST_logo.png'), style= {'align':'right','height':'20%', 'width':'20%'}),
    html.H6('Developed by Henrique Petrucci, 87202', style= {'text-align':'right','background-color': 'lavender', 'color': 'green'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Exploratory Data Analysis', value='tab-2'),
        dcc.Tab(label='Clustering', value='tab-3'),
        dcc.Tab(label='Feature Selection', value='tab-4'),
        dcc.Tab(label='Regression', value='tab-5'),
        
    ]),
    html.Div(id='tabs-content')   
])    
    
@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))


def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
        html.H4('Firstly, it is presented a table with all the data available as well as a graph where it can be seen the Power Consumption regarding the respective time. '),
             dcc.Dropdown(
        id='checkboxes',
        options=[
            {'label': 'Table ', 'value': 20},
            {'label': 'Graph', 'value': 21},
        ], 
        value=20
        ),

        html.Div(id='raw_html'),
 
        ]
    )                 
    elif tab == 'tab-2':
         return html.Div([
        html.H4('After exploring the raw data, it was decided to add some features (Hour, Weekday and Month) and the values with consumption under 60 kW were removed.'),
        html.H4('Again, it is presented the table with the data after EDA and the new graph with Power Consumptions.'),
            dcc.Dropdown(
        id='drops1',
        options=[
            {'label': 'Table ', 'value': 23},
            {'label': 'Graph', 'value': 24},
        ], 
        value=23
        ),

        html.Div(id='EDA_html'),      
        
       # dcc.Graph(
       #          id='yearly-data',
       #          figure={
       #              'data': [
       #                  {'x': data_clean.Date, 'y': data_clean.Power_kW, 'type': 'plot'}
       #              ],
       #              'layout': {
       #                  'title': 'Civil Building Power Consumption (kW)'
       #              }
       #          }
       #      )   
        ]
    )                 
    elif tab == 'tab-3':
        return html.Div([
            html.H4('Cluster analysis is started by choosing the number of clusters, followed by showing the graphs where the clusters are presented using different features and finally, as the objective is to forecast energy demand, a graph representing the 3 clusters in the available data in a one-day time frame is presented:'),
            dcc.Dropdown(
        id='dropdown',
        options=[
            {'label': 'Number of Clusters', 'value': 1},
            {'label': 'Clustering Graphics', 'value': 2},
            {'label': 'Final Clusters', 'value': 3}
        ], 
        value=1
        ),

        html.Div(id='clustering_html'),
        ])


    elif tab == 'tab-4':
         return html.Div([
            html.H4('The features corresponding to each number in the graphs are:'),
            html.H6('1- Temperature', style= {'color': 'grey'}),
            html.H6('2- Relative Humidity', style= {'color': 'grey'}),
            html.H6('3- Wind Speed', style= {'color': 'grey'}),
            html.H6('4- Wind Gust', style= {'color': 'grey'}),
            html.H6('5- Pressure', style= {'color': 'grey'}),
            html.H6('6- Solar Radiation', style= {'color': 'grey'}),
            html.H6('7- Precipitation', style= {'color': 'grey'}),
            html.H6('8- Hour', style= {'color': 'grey'}),
            html.H6('9- Weekday', style= {'color': 'grey'}),
            html.H6('10- Month', style= {'color': 'grey'}),
            html.H6('11- Power at Previous Hour', style= {'color': 'grey'}),
            html.H4('There are different Feature Selection Methods, as it can be found below:'),
            dcc.RadioItems(
        id='radio',
        options=[
            {'label': 'Filter Method: F-test', 'value': 4},
            {'label': 'Filter Method: Mutual Information', 'value': 5},
            {'label': 'Ensemble Method: Gradient Boosting ', 'value': 6},
            {'label': 'Ensemble Method: Adaptative Boosting ', 'value': 7},
            {'label': 'Wrapper Method: Sequential Forward Selection ', 'value': 8}
        ], 
        value=4
        ),
        html.Div(id='feature_html'),


        html.H5('Based on the graphs, the features selected were:'),
        html.H5('Solar Radiation, Hour, Weekday and Power at Previous Hour', style= {'background-color': 'lavender', 'color': 'green'}),
        ])
     
    elif tab == 'tab-5':
         return html.Div([
                html.H4('There were used different regression models:'),
                dcc.Dropdown(
                id='dropdown1',
        options=[
            {'label': 'Linear', 'value': 11},
            {'label': 'Support Vector', 'value': 12},
            {'label': 'Decision Tree', 'value': 13},
            {'label': 'Random Forest', 'value': 14},
            {'label': 'Light Gradient Boosting Machine', 'value': 15}
        ], 
        value=11
        ),

        html.Div(id='regression_html'),
        
        html.H5('Light Gradient Boosting Machine is the best regression model for this forecast model! ', style= {'background-color': 'lavender', 'color': 'green'})

        ])



@app.callback(Output('EDA_html', 'children'), 
              Input('drops1', 'value'))

def render_figure_html(EDA_drop):
        
    if EDA_drop == 24:
        return html.Div([
            dcc.Graph(
                 id='yearly-data1',
                 figure={
                     'data': [
                         {'x': data_exp.Date, 'y': data_exp.Power_kW, 'type': 'plot'}
                     ],
                     'layout': {
                         'title': 'Civil Building Power Consumption (kW)'
                     }
                 }
             ) 
            ])
    
    if EDA_drop == 23:
        return html.Div([dash_table.DataTable(
    id='table1',
    columns=[{"name": i, "id": i} for i in data_exp.columns],
    data=data_exp.to_dict('records'))
    
    ])

@app.callback(Output('raw_html', 'children'), 
              Input('checkboxes', 'value'))

def render_figure_html(raw_drop):
        
    if raw_drop == 21:
        return html.Div([
            dcc.Graph(
                 id='yearly-data',
                 figure={
                     'data': [
                         {'x': data_all.Date, 'y': data_all.Power_kW, 'type': 'plot'}
                     ],
                     'layout': {
                         'title': 'Civil Building Power Consumption (kW)'
                     }
                 }
             ) 
            ])
    
    if raw_drop == 20:
        return html.Div([dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in data_all.columns],
    data=data_all.to_dict('records'))
    
    ])





@app.callback(Output('clustering_html', 'children'), 
              Input('dropdown', 'value'))

def render_figure_html(clustering_drop):
    
    if clustering_drop == 1:
        return html.Div([html.Img(src=app.get_asset_url('NClusters.png')),
               html.H6('3 Clusters is a good choice!', style= {'background-color': 'lavender', 'color': 'green'})])
                
    elif clustering_drop == 2:
        return html.Div([html.Img(src=app.get_asset_url('TempvsPower.png')),
                         html.Img(src=app.get_asset_url('PowervsWeekday.png')),
                         html.Img(src=app.get_asset_url('PowervsHour.png')),
                         html.H6('The x-axis in these 3 graphs is Power (kW)'),
                         html.Img(src=app.get_asset_url('MonthvsTemp.png')),
                         html.H6('The x-axis in the graph is Month'),
                         html.Img(src=app.get_asset_url('FinalClusters1.png')),
                         html.H5('Finally, it is possible to represent the 3 Clusters in a 3D graph, where the x-axis is Hour, y-axis is Weekday and z-axis is Power (kW).', style= {'background-color': 'lavender', 'color': 'green'})])
                        
    elif clustering_drop == 3:
        return html.Div([html.Img(src=app.get_asset_url('FinalClusters2.png')),
                         html.H5('Mainly, there are Low, Medium and High Consumption Days. This difference is largely perceived between 6am and 8pm', style= {'background-color': 'lavender', 'color': 'green'}),
                         
                        ])
 
@app.callback(Output('feature_html', 'children'), 
              Input('radio', 'value'))

 
def render_figure_html(feature_drop):   
    
    if feature_drop == 4:
        return html.Div([html.Img(src=app.get_asset_url('f_regression.png'))])
             
                
    elif feature_drop == 5:
        return html.Div([html.Img(src=app.get_asset_url('mutual_info.png'))])
                        
                        
    elif feature_drop == 6:
        return html.Div([html.Img(src=app.get_asset_url('Gradient Boosting.png')),])

    elif feature_drop == 7:
        return html.Div([html.Img(src=app.get_asset_url('AdaBoost.png')),])

    elif feature_drop == 8:
        return html.Div([html.Img(src=app.get_asset_url('sfs.png')),
                         html.H3('Important Note:', style= {'background-color': 'WhiteSmoke','color': 'Red'}),
                         html.H6('In this last method, the features are numbered in 0 to 10, so the number of each features is x-1 being x the original feature number')])

@app.callback(Output('regression_html', 'children'), 
              Input('dropdown1', 'value'))

def render_figure_html(regression_drop):   
    
    if regression_drop == 11:
        return html.Div([html.H6('The left side graph presents the values predicted by each regression model (in yellow) vs the real values of Power consumption (in Blue) for a representative subset of data. It is important to notice that the blue line is the same for all models (meaning the subset used was always the same). '),
                         html.H6('The right side graph represents how far are the prediced values from the real ones. This can be assessed by analysing how far is each point from the straight line x=y.'),
                         html.H6('Below the graphs, there are the different errors for each regression model'),
                         html.Img(src=app.get_asset_url('LinearRegression1.png')),
                         html.Img(src=app.get_asset_url('LinearRegression2.png')),
                         html.H6('Mean Absolute Error: 18.998794114740853',style= {'color': 'grey'}),
                         html.H6('Mean Squared Error: 794.7581861703917',style= {'color': 'grey'}),
                         html.H6('Root-Mean Squared Error: 28.19145590724948 ',style= {'color': 'grey'}),
                         html.H6('Coefficient of Variation of Root-Mean Squared Error: 0.16039617359796796 ',style= {'color': 'grey'})

                         
                         
                         ])

    if regression_drop == 12:
        return html.Div([html.H6('The left side graph presents the values predicted by each regression model (in yellow) vs the real values of Power consumption (in Blue) for a representative subset of data. It is important to notice that the blue line is the same for all models (meaning the subset used was always the same). '),
                         html.H6('The right side graph represents how far are the prediced values from the real ones. This can be assessed by analysing how far is each point from the straight line x=y.'),
                         html.H6('Below the graphs, there are the different errors for each regression model'),
                         html.Img(src=app.get_asset_url('SupportVector1.png')),
                         html.Img(src=app.get_asset_url('SupportVector2.png')),
                         html.H6('Mean Absolute Error: 9.965202112085242 ',style= {'color': 'grey'}),
                         html.H6('Mean Squared Error: 254.18524253627976 ',style= {'color': 'grey'}),
                         html.H6('Root-Mean Squared Error: 15.943187966535419 ',style= {'color': 'grey'}),
                         html.H6('Coefficient of Variation of Root-Mean Squared Error: 0.09070926855281192 ',style= {'color': 'grey'})

                         
                         
                         ])  

    if regression_drop == 13:
        return html.Div([html.H6('The left side graph presents the values predicted by each regression model (in yellow) vs the real values of Power consumption (in Blue) for a representative subset of data. It is important to notice that the blue line is the same for all models (meaning the subset used was always the same). '),
                         html.H6('The right side graph represents how far are the prediced values from the real ones. This can be assessed by analysing how far is each point from the straight line x=y.'),
                         html.H6('Below the graphs, there are the different errors for each regression model'),
                         html.Img(src=app.get_asset_url('DecisionTree1.png')),
                         html.Img(src=app.get_asset_url('DecisionTree2.png')),
                         html.H6('Mean Absolute Error: 9.665758615049816  ',style= {'color': 'grey'}),
                         html.H6('Mean Squared Error: 281.29485254518056  ',style= {'color': 'grey'}),
                         html.H6('Root-Mean Squared Error: 16.771847022471334  ',style= {'color': 'grey'}),
                         html.H6('Coefficient of Variation of Root-Mean Squared Error: 0.09542395027151117 ',style= {'color': 'grey'})

                         
                         
                         ])          
    
    if regression_drop == 14:
        return html.Div([html.H6('The left side graph presents the values predicted by each regression model (in yellow) vs the real values of Power consumption (in Blue) for a representative subset of data. It is important to notice that the blue line is the same for all models (meaning the subset used was always the same). '),
                         html.H6('The right side graph represents how far are the prediced values from the real ones. This can be assessed by analysing how far is each point from the straight line x=y.'),
                         html.H6('Below the graphs, there are the different errors for each regression model'),
                         html.Img(src=app.get_asset_url('RandomForest1.png')),
                         html.Img(src=app.get_asset_url('RandomForest2.png')),
                         html.H6('Mean Absolute Error: 7.427195446189826   ',style= {'color': 'grey'}),
                         html.H6('Mean Squared Error: 160.88440771626918   ',style= {'color': 'grey'}),
                         html.H6('Root-Mean Squared Error: 12.684021748494015   ',style= {'color': 'grey'}),
                         html.H6('Coefficient of Variation of Root-Mean Squared Error: 0.0721661400172199 ',style= {'color': 'grey'})

                         
                         
                         ]) 
    
    if regression_drop == 15:
        return html.Div([html.H6('The left side graph presents the values predicted by each regression model (in yellow) vs the real values of Power consumption (in Blue) for a representative subset of data. It is important to notice that the blue line is the same for all models (meaning the subset used was always the same). '),
                         html.H6('The right side graph represents how far are the prediced values from the real ones. This can be assessed by analysing how far is each point from the straight line x=y.'),
                         html.H6('Below the graphs, there are the different errors for each regression model'),
                         html.Img(src=app.get_asset_url('LGBM1.png')),
                         html.Img(src=app.get_asset_url('LGBM2.png')),
                         html.H6('Mean Absolute Error: 7.317736359861035    ',style= {'color': 'grey'}),
                         html.H6('Mean Squared Error: 160.50736216723396    ',style= {'color': 'grey'}),
                         html.H6('Root-Mean Squared Error: 12.669150017551846    ',style= {'color': 'grey'}),
                         html.H6('Coefficient of Variation of Root-Mean Squared Error: 0.07208152683705105 ',style= {'color': 'grey'})

                         
                         
                         ]) 
                

if __name__ == '__main__':
    app.run_server(debug=True)
