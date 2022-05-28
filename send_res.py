import requests
import json

#                           url
def submit(results, url="https://tinyurl.com/IML2022"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['results']}")
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")

with open("JsonExampleResults/result.json", "r") as f:
    dictio = json.load(f)
    submit(dictio)


