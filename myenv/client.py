import psutil
import requests
import time

url = "http://13.232.212.55:8000/predict/"

CPU_THRESHOLD = 5   # in percentage
MEMORY_THRESHOLD = 50 

while True:
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            user_count = len(psutil.users())
            
            if cpu_usage > CPU_THRESHOLD or memory_usage > MEMORY_THRESHOLD:
                data = {
                    "cpu_usage": cpu_usage,                                                
                    "memory_usage": memory_usage,
                    "user_count": user_count
                }
            
                print(f"Sending Data to API: {data}")
                response = requests.post(url, json=data)
                if response.status_code == 200:
                    print(f"Prediction Received: {response.json()}")
                else:
                    print(f"Failed to get a valid response. Status Code: {response.status_code}")
            else:
                print(f"Conditions not met. CPU: {cpu_usage}%, Memory: {memory_usage}%")
        except requests.exceptions.RequestException as e:
            print(f"Error while sending request: {e}")
        time.sleep(300)

