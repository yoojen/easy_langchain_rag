from difflib import SequenceMatcher

# def __init__(self, CLOSING_PHRASES):
#     self.CLOSING_PHRASES = CLOSING_PHRASES


def is_conversation_closing(user_input: str, CLOSING_PHRASES) -> bool:
    """
    Check if the user input indicates the end of a conversation.

    This function evaluates whether the given user input contains any of the
    predefined closing phrases, suggesting that the user wants to close the conversation.

    Parameters
    ----------
    user_input : str
        The input string provided by the user.
    CLOSING_PHRASES : list
        A list of phrases that indicate the conversation is closing.

    Returns
    -------
    bool
        True if the user input contains a closing phrase, False otherwise.
    """

    user_input = user_input.strip().lower()
    for phrase in CLOSING_PHRASES:
        if phrase in user_input:
            return True
    return False


def fuzzy_match(user_input: str, CLOSING_PHRASES, threshold: float = 0.85) -> bool:
    """
    Perform a fuzzy match on the given user input against the closing phrases.

    This function measures the similarity between the user input and the given
    closing phrases using the SequenceMatcher from the difflib library. If the
    similarity ratio is equal to or exceeds the given threshold, the function
    returns True. Otherwise, False is returned.

    Parameters
    ----------
    user_input : str
        The input string provided by the user.
    CLOSING_PHRASES : list
        A list of phrases that indicate the conversation is closing.
    threshold : float, optional
        The minimum similarity ratio required to consider the input a closing phrase.
        Defaults to 0.85.

    Returns
    -------
    bool
        True if the user input is a fuzzy match for one of the closing phrases,
        False otherwise.
    """
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
