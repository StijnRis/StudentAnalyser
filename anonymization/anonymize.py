import re
from pathlib import Path
from typing import Dict

import pandas as pd
from presidio_analyzer import (
    AnalyzerEngine,
    LemmaContextAwareEnhancer,
    Pattern,
    PatternRecognizer,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from tqdm import tqdm

NLP_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [
        {"lang_code": "en", "model_name": "en_core_web_lg"},
        {"lang_code": "nl", "model_name": "nl_core_news_lg"},
    ],
}

# Load custom names from the external file
# Files to load (first is the original absolute path, second is the local banned names file)
paths = [
    Path(
        "C:/University/Honours/Data/StanislasExperiment1/Consent forms/names_of_accepted_consent_forms.txt"
    ),
    Path(
        "C:/University/Honours/Data/StanislasExperiment1/Consent forms/studentnummers_of_accepted_consent_forms.txt"
    ),
    Path("C:/University/Honours/StudentAnalyser/anonymization/data/banned_persons.txt"),
]

names = []
for p in paths:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            for token in line.strip().split():
                if token:
                    names.append(token)


# Deduplicate and escape for safe use inside a regex; sort by length to prefer longer matches first
seen = set()
CUSTOM_NAMES_LIST = []
for n in sorted(names, key=len, reverse=True):
    if n not in seen:
        seen.add(n)
        CUSTOM_NAMES_LIST.append(re.escape(n))

with open("anonymization/data/banned_words.txt", "r", encoding="utf-8") as f:
    BANNED_WORDS = [line.strip() for line in f if line.strip()]

with open("anonymization/data/banned_locations.txt", "r", encoding="utf-8") as f:
    BANNED_LOCATIONS = [line.strip() for line in f if line.strip()]


def anonymize(data: Dict[str, pd.DataFrame]):
    # Names recognizer using the custom names list
    custom_names_recognizer = PatternRecognizer(
        name="CustomDenyListRecognizer",
        supported_entity="PERSON",
        patterns=[
            Pattern(
                regex=r"(" + "|".join(CUSTOM_NAMES_LIST) + r")",
                name="CustomNamePattern",
                score=1.0,
            )
        ],
    )

    # Recogniser for banned words
    banned_words_recognizer = PatternRecognizer(
        name="BannedWordsRecognizer",
        supported_entity="BANNED_WORD",
        patterns=[
            Pattern(
                regex=r"(" + "|".join(re.escape(word) for word in BANNED_WORDS) + r")",
                name="BannedWordPattern",
                score=1.0,
            )
        ],
    )

    # Recogniser for banned locations
    banned_locations_recognizer = PatternRecognizer(
        name="BannedLocationsRecognizer",
        supported_entity="LOCATION",
        patterns=[
            Pattern(
                regex=r"("
                + "|".join(re.escape(loc) for loc in BANNED_LOCATIONS)
                + r")",
                name="BannedLocationPattern",
                score=1.0,
            )
        ],
    )

    # Recognizer for Dutch postcodes: 4 digits optionally followed by up to 2 letters
    postcode_recognizer = PatternRecognizer(
        name="PostcodeRecognizer",
        supported_entity="POSTCODE",
        context=[
            "postcode",
            "post code",
            "zip code",
            "zip",
            "adres",
            "straat",
            "woonplaats",
        ],
        patterns=[
            Pattern(
                regex=r"(\d{4})",
                name="DutchPartialPostcode",
                score=0.01,
            ),
            Pattern(
                regex=r"(\d{4}\s?[A-Za-z]{2})",
                name="DutchFullPostcode",
                score=0.8,
            ),
        ],
    )

    # Recognizer for huisnummer (house number): 1-5 digits with an optional trailing letter
    huisnummer_recognizer = PatternRecognizer(
        name="HuisnummerRecognizer",
        supported_entity="HOUSENUMBER",
        context=[
            "huisnummer",
            "house number",
            "address number",
            "adres",
            "straat",
            "housenumber",
            "wonen",
            "woont",
            "living at",
        ],
        patterns=[
            Pattern(
                regex=r"\b\d{1,5}[A-Za-z]?\b",
                name="HuisnummerPattern",
                score=0.01,
            )
        ],
    )

    age_recognizer = PatternRecognizer(
        name="AgeDetectorRecognizer",
        supported_entity="AGE",
        context=[
            "age",
            "years old",
            "yrs",
            "jaar",
            "jaar oud",
            "leeftijd",
        ],
        patterns=[
            Pattern(
                regex=r"(?i)\b([1-9][0-9]?|1[01][0-9])\b",
                name="OnlyDigitsAgePattern",
                score=0.01,
            ),
            Pattern(
                regex=r"(?i)\b([1-9][0-9]?|1[01][0-9])\s*(?:years?|yrs?|y\.?|jaar(?:\s*oud)?|jr\.?|j\.?)\b",
                name="AgePattern",
                score=0.5,
            )
        ],
    )

    # Create the NLP Engine Provider with multi-language config
    provider = NlpEngineProvider(nlp_configuration=NLP_CONFIG)
    nlp_engine = provider.create_engine()

    context_enhancer = LemmaContextAwareEnhancer(
        context_prefix_count=10, context_suffix_count=10
    )

    # Initialize the AnalyzerEngine with the multi-language engine
    analyzer_engine = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en", "nl"],
        context_aware_enhancer=context_enhancer,
        default_score_threshold=0.3,
    )
    analyzer_engine.registry.add_recognizer(custom_names_recognizer)
    analyzer_engine.registry.add_recognizer(banned_words_recognizer)
    analyzer_engine.registry.add_recognizer(banned_locations_recognizer)
    analyzer_engine.registry.add_recognizer(postcode_recognizer)
    analyzer_engine.registry.add_recognizer(huisnummer_recognizer)
    analyzer_engine.registry.add_recognizer(age_recognizer)

    anonymizer = AnonymizerEngine()

    seen_cache: Dict[str, str] = {}

    entities = [
        "AGE",
        "BANNED_WORD",
        "DATE_TIME",
        "EMAIL_ADDRESS",
        "HOUSENUMBER",
        "IBAN_CODE",
        "IP_ADDRESS",
        "LOCATION",
        "NRP",
        "PERSON",
        "PHONE_NUMBER",
        "POSTCODE",
        "URL",
    ]

    def anonymize_text(text: str, pbar: tqdm) -> str:
        # Advance the progress bar for each invocation (even if cached)
        pbar.update(1)

        # Skip non-strings
        if not isinstance(text, str):
            return text

        # Skip excessively large texts
        if len(text) > 3000:
            return "<TEXT_TOO_LONG_TO_ANONYMIZE>"

        # Return cached result if available
        if text in seen_cache:
            return seen_cache[text]

        nl_results = analyzer_engine.analyze(
            text=text, language="nl", entities=entities
        )
        en_results = analyzer_engine.analyze(
            text=text, language="en", entities=entities
        )
        combined_results = nl_results + en_results
        anonymized = anonymizer.anonymize(text=text, analyzer_results=combined_results)
        result_text = anonymized.text

        # Cache the anonymized value so we don't re-run analysis for identical texts
        seen_cache[text] = result_text

        return result_text

    def anonymize_dataframe(dataframe: str, column: str):
        pbar = tqdm(
            total=len(data[dataframe]), desc=f"Anonymizing {dataframe} {column}"
        )
        data[dataframe][column] = data[dataframe][column].apply(
            lambda x: anonymize_text(x, pbar)
        )

    anonymize_dataframe("file_versions", "code")
    anonymize_dataframe("execution_outputs", "output_text")
    anonymize_dataframe("execution_errors", "traceback")
    anonymize_dataframe("edits", "filename")
    anonymize_dataframe("edits", "selection")
    data["users"].drop(columns=["username", "group"], inplace=True)
    anonymize_dataframe("messages", "body")
