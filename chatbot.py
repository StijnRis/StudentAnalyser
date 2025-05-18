import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

cache_path = "output/chatbot_cache.json"
cache: dict[str, str] = {}

server = os.getenv("OPEN_WEB_UI_SERVER")
if server is None:
    raise ValueError("OPEN_WEB_UI_SERVER is not set")

key = os.getenv("OPEN_WEB_UI_API_KEY")
if key is None:
    raise ValueError("OPEN_WEB_UI_API_KEY is not set")

url = f"{server}/api/chat/completions"


def save_cache():
    global cache
    with open(cache_path, "w") as file:
        file.write(json.dumps(cache))


def load_cache():
    global cache
    if os.path.exists(cache_path):
        with open(cache_path, "r") as file:
            cache = json.load(file)


load_cache()

def ask_question(question):
    global cache

    if question in cache:
        return cache[question]

    return ask_question_without_cache(question)


def ask_question_without_cache(question):
    print("Asking question without cache")
    global cache

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "llama3.2:latest",
        "messages": [{"role": "user", "content": question}],
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise ValueError(f"Error: {response.status_code} - {response.text}")

    cache[question] = response.json()["choices"][0]["message"]["content"]

    # Save cache to file every 10 questions
    if len(cache) % 10 == 0:
        print(f"{len(cache)} questions in cache")
    save_cache()

    return cache[question]
