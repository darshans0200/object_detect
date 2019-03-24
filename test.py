#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:26:07 2019

@author: sankalp
"""
print ("Content-Type: application/json\r\n")
import subprocess
import os
import cgi, cgitb
import sys
import logging
from shlex import split

"""
logging.basicConfig(filename='/home/sankalp/example.log',level=logging.DEBUG)
"""
form = cgi.FieldStorage()
from urllib.parse import parse_qs
params = parse_qs(os.environ['QUERY_STRING'])
image = params.get('image', [None])[0]
storeId = params.get('id', [None])[0]
fridgeId = params.get('id2', [None])[0]
print(image,storeId,fridgeId)

"""
to execute from command line

image = sys.argv[1]
storeId = sys.argv[2]
fridgeId = sys.argv[3]
print(image,storeId,fridgeId)

"""
cmd = 'bash /home/sankalp/./conda_act.sh {} {} {}'.format(image, storeId, fridgeId)
res = subprocess.check_output(split(cmd))
print(res)

