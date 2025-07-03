from difflib import SequenceMatcher

# def __init__(self, CLOSING_PHRASES):
#     self.CLOSING_PHRASES = CLOSING_PHRASES


def is_conversation_closing(user_input: str, CLOSING_PHRASES) -> bool:
    user_input = user_input.strip().lower()
    for phrase in CLOSING_PHRASES:
        if phrase in user_input:
            return True
    return False


def fuzzy_match(user_input: str, CLOSING_PHRASES, threshold: float = 0.85) -> bool:
    user_input = user_input.lower().strip()
    for phrase in CLOSING_PHRASES:
        similarity = SequenceMatcher(None, user_input, phrase).ratio()
        if similarity >= threshold:
            return True
    return False


def detect_closing_intent(user_input: str, CLOSING_PHRASES) -> bool:
    """
    Detect if user input is a closing intent.

    A closing intent is a phrase that indicates the user is done with the conversation.
    This function checks if the user input contains any of the phrases in
    `CLOSING_PHRASES` or if the user input is similar to one of the phrases in
    `CLOSING_PHRASES` (using fuzzy matching).

    Parameters
    ----------
    user_input : str
        The user input to check.

    Returns
    -------
    bool
        True if the user input is a closing intent, False otherwise.
    """
    if is_conversation_closing(user_input, CLOSING_PHRASES):
        return True
    if fuzzy_match(user_input, CLOSING_PHRASES):
        return True
    return False


def token_counter():
    pass


def trim_messages():
    pass
