# %%[markdown]
#
# # Working with API and JSON in Python
#
# We will make API request calls using `requests.get()` function.
# Instead of the web server serving us (the client browser) the html codes/pages,
# this time api call **usually** will have the server returning some data, and **often** in
# the form of JavaScript Object Notation (JSON) object.
#
# To python, a JSON object is nothing more than just some
# dictionary of dictionary of list of ...
# When something is referenced by some keywords, it will be stored as key => value in
# a dictionary. If there are similar things stored together, it will be as a list.
#
# Example: myHonda = {'color': 'red', 'wheels': 4, 'engine': {'cylinders': 4, 'size': 2.2} }
# Example: myGarage = [ {'VIN':'abcd', 'car': myHonda } , {'VIN':'fghi', 'car': myFiat} ]
# For the last example, when things is a list (also to a certain extend, for
# dictionaries as well), it will be much easier to name them using plural nouns. Avoid verbs.
# For example: myCars = [ {'VIN':'abcd', 'car': myHonda } , {'VIN':'fghi', 'car': myFiat} ]
# (it is nested, list of dictionary of ...) then the codes to loop thru will be like:
# for myCar in myCars : print(myCar['VIN'])
# Much easier to understand and follow.
#

# %%
# import json
# %pip install requests
from cytoolz.dicttoolz import merge
import pandas as pd
import json
import requests # This is a new package
from requests.api import head
# from time import sleep  # sometimes, you want to slow down your API calls. 
# For example, some providers limit the frequency of requests

# %%
# Basic
# Let us find out where the International Space Station is currently
#
response = requests.get("http://api.open-notify.org/iss-now.json")
status_code = response.status_code
print(f'status_code: {status_code}\n')  # 200 (OK)

# Headers is a dictionary
print(f'headers: {response.headers}\n')
# Notice that the header is case-InSensiTiVe type
# Both lines below are the same
print(response.headers['content-Type'])
print(response.headers['content-type'])

# %% [markdown]
# The list of status codes:
#
# * 200 — Everything went okay, and the server returned a result (if any).
# * 301 — The server is redirecting you to a different endpoint. This can happen when a company switches domain names, or when an endpoint's name has changed.
# * 401 — The server thinks you're not authenticated. This happens when you don't send the right credentials to access an API.
# * 400 — The server thinks you made a bad request. This can happen when you don't send the information that the API requires to process your request (among other things).
# * 403 — The resource you're trying to access is forbidden, and you don't have the right permissions to see it.
# * 404 — The server didn't find the resource you tried to access.
#
# This "endpoint" does not need authentication, nor any other parameteres. So the requests
# call all went fine.
#

# %%
# We can now parse the response content to find info we need:
print(f'content type: {type(response.content)}')
print('content:')
print(response.content)  # 'b' indiates it is byte type.

# %%
# The content here is of JSON format. We can load it using:
print('JSON :')
print(response.json())

# %%
# Instead of using response.json(), the equivalent (longer version) will be
jsondata2 = json.loads(response.content)
# The opposite of json.loads() (from string to JSON) is json.dumps() (from JSON to string)
# Also, json.loads(string) will convert the string to json, while json.load(file_path)
# will convert the content in the file to json.
print(jsondata2)

print(response.json() == jsondata2)  # True # identical info
print(response.json() is jsondata2)  # False # these two are not shallow copies

# %%
# Examples of getting different status codes:
# 404, not found; wrong endpoint; read API doc
print(requests.get('http://api.open-notify.org/iss-pass').status_code)
# 400, bad reqeusts; read API doc to learn how to use this endpoint
print(requests.get('http://api.open-notify.org/iss-pass.json').status_code)

# %%
# Passing parameters
parameters = {"description": "pets"}
response = requests.get(
    "https://api.publicapis.org/entries", params = parameters)
print(f'status_code: {response.status_code}\n')  # 200, success
jsondata = json.loads(response.content) #response.json()
print(jsondata)
#
# This gets the same data as the command above
response = requests.get(
    "https://api.publicapis.org/entries?description=pets&title=pets")
# You can actually also paste this "url" in a browser directly.
#

# %%
# Many api access requires either authentication or key.
# Try the GitHub API
# for GitHub: sushovan4
mygitheaders = {
    "Authorization": "Token [TOKEN]"}
response = requests.get("https://api.github.com/user", headers = mygitheaders)
# In general, we can combine:
# response = requests.get("https://api.regression.fit/endpt.json", headers=mygitheaders, params=parameters)
user = response.json()
print(user)
# The endpoint here is 'user'