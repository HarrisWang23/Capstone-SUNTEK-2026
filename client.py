import os
import requests
import json
from litellm import completion

# Configuration
os.environ["LITELLM_PROXY_API_KEY"] = "sk-xxxxx"
os.environ["LITELLM_PROXY_API_BASE"] = "https://litellm-gke-test.suntekcorps.com"
api_key = os.environ["LITELLM_PROXY_API_KEY"]
api_base = os.environ["LITELLM_PROXY_API_BASE"]
# The endpoint to list models is typically /models or /v1/models
url = f"{api_base}/models"
headers = {
    "Authorization": f"Bearer {api_key}"
}
try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    models_data = response.json()
    print(json.dumps(models_data, indent=2))
    # If you want to list just the model IDs:
    # for model in models_data.get('data', []):
    #     print(model['id'])
except requests.exceptions.RequestException as e:
    print(f"Error fetching models: {e}")

# LiteLLM proxy completion call
messages = [{"content": "Hello, how are you?", "role": "user"}]

response = completion(
    model="litellm_proxy/bltcy-default_claude-haiku-4-5-20251001",
    messages=messages,
)
print(response.choices[0].message.content)
