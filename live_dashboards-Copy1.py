# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:04:48 2019

@author: win8
"""

import dash
from dash.dependencies import Output, Events
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque

X = deque(maxlen=20)
Y = deque(maxlen=20)
X.append(1)
Y.append(1)

app = dash.Dash(__name__)
app.layout()
