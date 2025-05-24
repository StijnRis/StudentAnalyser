import asyncio
import json
import os

import aiohttp
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR")
if OUTPUT_DIR is None:
    raise ValueError("OUTPUT_DIR is not set")

SERVER = os.getenv("OPEN_WEB_UI_SERVER")
if SERVER is None:
    raise ValueError("OPEN_WEB_UI_SERVER is not set")

SERVER_KEY = os.getenv("OPEN_WEB_UI_API_KEY")
if SERVER_KEY is None:
    raise ValueError("OPEN_WEB_UI_API_KEY is not set")

cache_path = f"{OUTPUT_DIR}/chatbot_cache.json"
cache: dict[str, str] = {}

url = f"{SERVER}/api/chat/completions"


def save_cache():
    global cache
    with open(cache_path, "w") as file:
        file.write(json.dumps(cache))


def load_cache():
    global cache
    if os.path.exists(cache_path):
        with open(cache_path, "r") as file:
            cache = json.load(file)
    print(f"Loaded {len(cache)} questions from cache")


load_cache()


async def ask_question_async(question, session):
    global cache

    if question in cache:
        return cache[question]

    return await ask_question_without_cache_async(question, session)

async def ask_question_without_cache_async(question, session):
    print("Asking question to chatbot (async)")
    start_time = asyncio.get_event_loop().time()
    global cache

    headers = {
        "Authorization": f"Bearer {SERVER_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gemma3:27b",
        "messages": [{"role": "user", "content": question}],
    }

    async with session.post(url, headers=headers, json=data) as response:
        if response.status != 200:
            text = await response.text()
            raise ValueError(f"Error: {response.status} - {text}")
        result = await response.json()
    response_text = result["choices"][0]["message"]["content"]

    end_time = asyncio.get_event_loop().time()
    print(f"Response time: {end_time - start_time:.2f} seconds ({len(response_text)} characters)")

    # Save to cache
    cache[question] = response_text
    save_cache()

    return cache[question]

def add_column_through_chatbot(df, column_name, prompt_fn, extract_fn, max_retries=3):
    """
    Adds extracted answer of chatbot to the DataFrame using async requests, but is itself synchronous.
    """

    async def ask_with_retries_async(row, session):
        prompt = prompt_fn(row)
        last_response = None
        for i in range(max_retries):
            if i == 0:
                # Use sync ask_question_async for first try
                response = await ask_question_async(prompt, session)
            else:
                response = await ask_question_without_cache_async(prompt, session)
            last_response = response
            try:
                value = extract_fn(response, row)
                return prompt, response, value
            except Exception as e:
                last_error = str(e)
        return prompt, last_response, None

    async def process_all(rows):
        sem = asyncio.Semaphore(2)
        connector = aiohttp.TCPConnector()
        session = aiohttp.ClientSession(connector=connector)

        async def sem_task(row, session):
            async with sem:
                return await ask_with_retries_async(row, session)

        tasks = [sem_task(row, session) for row in rows]
        return await asyncio.gather(*tasks)

    rows = list(df.to_dict(orient="records"))
    results = asyncio.run(process_all(rows))
    df[column_name] = [r[0] for r in results]
    df[column_name + "_prompt"] = [r[0] for r in results]
    df[column_name + "_response"] = [r[1] for r in results]
    df[column_name] = [r[2] for r in results]
    return df
