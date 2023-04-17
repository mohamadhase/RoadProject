import re
import arabicstopwords.arabicstopwords as stp


def remove_phone_numbers(text: str) -> str:
    """
    Removes phone numbers from text.
    Args:
        text (str): The input text with phone numbers.
    Returns:
        str: The input text with phone numbers removed.
    """
    # regex pattern to match phone numbers
    pattern = r'(\+?\d{2,4}[ -]?)?\d{9,10}'
    # replace phone numbers with empty string
    return re.sub(pattern, '', text)


def remove_non_arabic_characters(text: str) -> str:
    """
    Removes non-Arabic characters from text.
    Args:
        text (str): The input text with non-Arabic characters.
    Returns:
        str: The input text with non-Arabic characters removed.
    """
    # regex pattern to match non-Arabic characters
    pattern = r'[^\u0600-\u06FF\s]+'
    # replace non-Arabic characters with empty string
    return re.sub(pattern, '', text)


def remove_end_of_line_and_tab_characters(text: str) -> str:
    """
    Removes end-of-line and tab characters from text.
    Args:
        text (str): The input text with end-of-line and tab characters.
    Returns:
        str: The input text with end-of-line and tab characters removed.
    """
    # regex pattern to match end-of-line and tab characters
    pattern = r'[\r\n\t]+'
    # replace end-of-line and tab characters with empty string
    return re.sub(pattern, '', text)


def clean_text(text: str) -> str:
    """
    Cleans text by removing phone numbers, non-Arabic characters, and end-of-line/tab characters.
    Args:
        text (str): The input text to be cleaned.
    Returns:
        str: The cleaned text.
    """
    # remove phone numbers from text
    clean_text = remove_phone_numbers(text)
    # remove non-Arabic characters from text
    clean_text = remove_non_arabic_characters(clean_text)
    # remove end-of-line and tab characters from text
    clean_text = remove_end_of_line_and_tab_characters(clean_text)
    # remove stop words from text
    clean_text = remove_arabic_stop_words(clean_text)
    # remove any numbers from text
    clean_text = remove_any_numbers(clean_text)
    return clean_text


def remove_arabic_stop_words(text: str) -> str:
    """Remove Arabic stop words from a given text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with Arabic stop words removed.

    """
    # Split the text into individual words and filter out the stop words
    remove_stop_words = [word for word in text.split() if not stp.is_stop(word)]
    # Join the remaining words back into a single string
    return " ".join(remove_stop_words)

def remove_any_numbers(text: str) -> str:
    """Remove any numerical digits from a given text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with numerical digits removed.

    """
    # Replace any sequence of numerical digits with an empty string
    return re.sub(r'\d+', '', text)