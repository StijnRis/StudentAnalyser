import asyncio
import json
import os
from typing import Any, Callable

import aiohttp
import pandas as pd
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

    timeout = aiohttp.ClientTimeout(total=300)
    try:
        async with session.post(
            url, headers=headers, json=data, timeout=timeout
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise ValueError(f"Error: {response.status} - {text}")
            result = await response.json()
    except asyncio.TimeoutError:
        print("Request timed out, retrying...")
        await asyncio.sleep(60)
        return await ask_question_without_cache_async(question, session)
    response_text = result["choices"][0]["message"]["content"]

    end_time = asyncio.get_event_loop().time()
    print(
        f"Response time: {end_time - start_time:.2f} seconds ({len(response_text)} characters)"
    )

    # Save to cache
    cache[question] = response_text
    save_cache()

    return cache[question]


def add_column_through_chatbot(
    df: pd.DataFrame,
    column_name: str,
    generate_prompt_fn: Callable[[pd.Series], str],
    extract_data_fn: Callable[[str, pd.Series], Any],
    default_value: Any,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Adds extracted answer of chatbot to the DataFrame using async requests, but is itself synchronous.
    """

    async def ask_with_retries_async(row, session):
        prompt = generate_prompt_fn(row)
        last_response = None
        for i in range(max_retries):
            if i == 0:
                response = await ask_question_async(prompt, session)
            else:
                response = await ask_question_without_cache_async(prompt, session)
            last_response = response
            try:
                value = extract_data_fn(response, row)
                return prompt, response, value
            except Exception as e:
                last_error = str(e)
        return prompt, last_response, default_value

    async def process_all(rows):
        sem = asyncio.Semaphore(2)
        connector = aiohttp.TCPConnector()
        session = aiohttp.ClientSession(connector=connector)

        async def sem_task(row, session):
            async with sem:
                return await ask_with_retries_async(row, session)

        tasks = [sem_task(row, session) for row in rows]
        result = await asyncio.gather(*tasks)
        await session.close()
        return result

    rows = list(df.to_dict(orient="records"))
    results = asyncio.run(process_all(rows))
    df[column_name] = [r[0] for r in results]
    df[column_name + "_prompt"] = [r[0] for r in results]
    df[column_name + "_response"] = [r[1] for r in results]
    df[column_name] = [r[2] for r in results]
    return df
