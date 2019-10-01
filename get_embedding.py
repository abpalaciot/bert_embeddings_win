# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:47:23 2019

@author: XZ935UB
"""

import requests

def get_embedding(txt):
    headers = {
        'Content-type': 'application/json',
    }
    
    data = '{"text":"'+txt+'"}'
    
    response = requests.post('http://localhost:5000/todo/api/v1.0/tasks', headers=headers, data=data)
    
    text=response.json()
    text=text['task']
    return text


text=get_embedding('hello world')
