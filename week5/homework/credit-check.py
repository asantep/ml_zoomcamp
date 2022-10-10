import requests


url = 'http://localhost:9898/credit'


client = {
    "reports": 0,
    "share": 0.245,
    "expenditure": 3.438,
    "owner": "yes"
}


response = requests.post(url=url, json=client).json()
print(response)