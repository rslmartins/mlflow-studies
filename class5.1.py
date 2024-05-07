import requests

url = "http://127.0.0.1:5001/invocations"

data = {"columns": ["size","year","garage"], "data": [[159.0, 2003, 2]]}

header = {"Content-Type": "application/json"}

response = requests.post(url, json=data, headers=header)

print(response)
print(response.text)