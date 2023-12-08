import requests

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(input_text):
    payload = {
        "inputs": input_text,
        "max_length": 128
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    print(response.json())
    return response.json()


# output = query("Generate a weather report")[0]["generated_text"]
#
# print(output)