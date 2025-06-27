from difflib import SequenceMatcher
from data import CLOSING_PHRASES

# def __init__(self, CLOSING_PHRASES):
#     self.CLOSING_PHRASES = CLOSING_PHRASES


def is_conversation_closing( user_input: str) -> bool:
    user_input = user_input.strip().lower()
    for phrase in CLOSING_PHRASES:
        if phrase in user_input:
            return True
    return False


def fuzzy_match( user_input: str, threshold: float = 0.85) -> bool:
    user_input = user_input.lower().strip()
    for phrase in CLOSING_PHRASES:
        similarity = SequenceMatcher(None, user_input, phrase).ratio()
        if similarity >= threshold:
            return True
    return False


def detect_closing_intent(user_input: str) -> bool:
    if is_conversation_closing(user_input):
        return True
    if fuzzy_match(user_input):
        return True
    return False


def token_counter():
    pass


def trim_messages():
    pass
