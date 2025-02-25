import requests

url = "http://127.0.0.1:8080/predict"
data = {"age": 39, "workclass": "Private", "education": "Bachelors"}  # Ajusta según tu modelo

response = requests.post(url, json=data)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")