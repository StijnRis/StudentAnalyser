import re
from typing import Dict

import pandas as pd

from langdetect import detect


def add_code_in_message(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add a boolean column 'contains_code' to the messages DataFrame indicating if the message contains code.
    """

    def contains_code(body: str) -> bool:
        import re

        if not isinstance(body, str):
            return False
        # Check for function calls or assignments
        if re.search(r"\b(print|input|float|int|str|len)\(", body):
            return True
        # Check for variable assignments ('variable = something')
        if re.search(r"\b\w+\s*=\s*[^=\n]+", body):
            return True
        # check for if statements
        if re.search(r"\bif\s+.*\s*:\s*", body):
            return True
        # Look for other common Python syntax
        if (
            re.search(r"\bdef\s+\w+\s*\(.*\):", body)
            or re.search(r"\bclass\s+\w+\s*\(.*\):", body)
            or re.search(r"\bimport\s+\w+", body)
            or re.search(r"\bfrom\s+\w+\s+import\s+\w+", body)
        ):
            return True
        return False

    data["messages"]["contains_code"] = data["messages"]["body"].apply(contains_code)


def add_message_length(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add a column 'message_length' to the messages DataFrame indicating the length of the message body.
    """
    data["messages"]["message_length"] = data["messages"]["body"].str.len()


def add_included_code_snippets(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add a column 'included_code_snippets' to the messages DataFrame with code snippets found in the message body.
    """

    def get_included_code_snippets(body: str):
        if not isinstance(body, str):
            return []
        codes = []
        matches = re.finditer(r"(\`\`\`python|\`)((.|\n)+?)\`{1,3}", body, re.DOTALL)
        for match in matches:
            code_snippet = match.group(2)
            codes.append(code_snippet.strip())
        return codes

    data["messages"]["included_code_snippets"] = data["messages"]["body"].apply(
        get_included_code_snippets
    )


def add_message_language(data: Dict[str, pd.DataFrame]) -> None:
    """
    Add a column 'language' to the messages DataFrame indicating the detected language of the message body.
    """

    def get_language(body: str) -> str:
        if not isinstance(body, str) or body == "":
            return "unknown"
        try:
            return detect(body)
        except Exception:
            return "unknown"

    data["messages"]["language"] = data["messages"]["body"].apply(get_language)

