import datetime
import json
import os
import re
import sys
import arabic_reshaper
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from fuzzywuzzy import fuzz
# from wordcloud import WordCloud
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(os.path.abspath(parent_path))
from constants import QUESTION_WORDS, HAWAJEZ, INFORMATION_WORDS
from collections import Counter



def is_question_arabic(text):
    """
    Check if the given text is a question in Arabic.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the text is a question in Arabic, False otherwise.
    """
    return any(fuzz.partial_ratio(word, text) >= 80 for word in QUESTION_WORDS)


def is_it_talk_about_hajez(text):
    """
    Check if the given text is talking about "hajez".

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the text is talking about "hajez", False otherwise.
    """
    return any(fuzz.partial_ratio(word, text) >= 70 for word in HAWAJEZ)


def is_it_give_information(text):
    """
    Check if the given text is giving information about "hajez".

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the text is giving information about "hajez", False otherwise.
    """
    return any(fuzz.partial_ratio(word, text) >= 70 for word in INFORMATION_WORDS)
from fuzzywuzzy import fuzz

def is_talking_about_it(word:str, text:str)->int:
    """
    Check if the given word is mentioned in the given text with high enough similarity using fuzzywuzzy.

    Args:
    - word: a string representing the word to check.
    - text: a string representing the text to search in.

    Returns:
    - 1 if the given word is mentioned in the given text with high enough similarity, 0 otherwise.
    """
    # using fuzzywuzzy if its fuzzy enough
    return 1 if fuzz.partial_ratio(word, text) >= 71 else 0


def is_it_match_status(keys:tuple, text:str)->int:
    """
    Check if any of the given words in a tuple is mentioned in the given text with high enough similarity using fuzzywuzzy.

    Args:
    - keys: a tuple of words to check if they are mentioned in the text.
    - text: a string representing the text to search in.

    Returns:
    - 1 if any of the given words in a tuple is mentioned in the given text with high enough similarity, 0 otherwise.
    """
    # Extract the relevant text from the input text by removing the text before the "|||" separator
    text = text.split("|||")[-1]
    # Check if any of the given words in a tuple is mentioned in the extracted text with high enough similarity using fuzzywuzzy
    return 1 if any(fuzz.partial_ratio(word, text) >= 70 for word in keys) else 0


def is_punct(word):
    """
    Check if a word consists entirely of punctuation characters.
    
    Args:
        word (str): The word to be checked.
        
    Returns:
        bool: True if the word is composed only of punctuation characters, False otherwise.
    """
    punct_chars = set('؟،؛?!.,:;')
    return all(char in punct_chars for char in word)


def combine_sub_words(output):
    """
    Combine sub-words in a list of output items.
    
    Args:
        output (list): The list of output items to process.
        
    Returns:
        list: The modified list with sub-words combined.
    """
    for index in range(len(output)):
        if index < len(output) - 1:
            if '##' in output[index+1]['word']:
                output[index]['word'] = output[index]['word'] + output[index+1]['word']
                output[index]['word'] = output[index]['word'].replace('##', '')
                # Delete the next word
                output.pop(index+1)
    return output



def word2features(sent, pos, i):
    """
    Generate a feature dictionary for a given word and its context.

    Args:
        sent (list): The list of words in a sentence.
        pos (list): The list of part-of-speech tags corresponding to the words in the sentence.
        i (int): The index of the current word.

    Returns:
        dict: A dictionary containing the generated features for the word.
    """
    word = sent[i]
    features = {
        'bias': 1.0,
        'word': word,
        'word.ispunct()': is_punct(word),
        'word.isnumeric()': word.isnumeric(),
        'word.isdigit()': word.isdigit(),
        'word.isalpha()': word.isalpha(),
        'word_len': len(word),
        'word_prefix1': word[:1],
        'word_prefix2': word[:2],
        'word_prefix3': word[:3],
        'word_prefix4': word[:4],
        'word_suffix1': word[-1:],
        'word_suffix2': word[-2:],
        'word_suffix3': word[-3:],
        'word_suffix4': word[-4:],
        'word_pos': pos[i],
    }
    if i > 0:
        prev_word = sent[i-1]
        features.update({
            'prev_word': prev_word,
            'prev_word.ispunct()': is_punct(prev_word),
            'prev_word.isnumeric()': prev_word.isnumeric(),
            'prev_word.isdigit()': prev_word.isdigit(),
            'prev_word.isalpha()': prev_word.isalpha(),
            'prev_word_suffix1': prev_word[-1:],
            'prev_word_suffix2': prev_word[-2:],
            'prev_word_suffix3': prev_word[-3:],
            'prev_word_suffix4': prev_word[-4:],
            'prev_word_pos': pos[i-1],
        })
    if i < len(sent) - 1:
        next_word = sent[i+1]
        features.update({
            'next_word': next_word,
            'next_word.ispunct()': is_punct(next_word),
            'next_word.isnumeric()': next_word.isnumeric(),
            'next_word.isdigit()': next_word.isdigit(),
            'next_word.isalpha()': next_word.isalpha(),
            'next_word_prefix1': next_word[:1],
            'next_word_prefix2': next_word[:2],
            'next_word_prefix3': next_word[:3],
            'next_word_prefix4': next_word[:4],
            'next_word_pos': pos[i+1]
        })
    return features


  

def group_locations(data):
    """
    Group locations based on their frequency and select the most frequent status for each location.

    Args:
        data (list): The list of dictionaries representing the data with 'location' and 'status' keys.

    Returns:
        list: The list of dictionaries containing the most frequent status for each location.
    """
    # Create a dictionary to store the frequency of each (location, status) combination
    frequency_dict = {}

    # Count the frequency of each (location, status) combination
    for item in data:
        location = item['location']
        status = item['status']
        key = (location, status)
        frequency_dict[key] = frequency_dict.get(key, 0) + 1

    # Create a dictionary to store the most frequent status for each location
    result_dict = {}

    # Find the most frequent status for each location
    for (location, status), frequency in frequency_dict.items():
        if location in result_dict:
            current_frequency, current_status = result_dict[location]
            if frequency > current_frequency:
                result_dict[location] = (frequency, status)
            elif frequency == current_frequency and data.index({'location': location, 'status': status}) > data.index({'location': location, 'status': current_status}):
                result_dict[location] = (frequency, status)
        else:
            result_dict[location] = (frequency, status)

    # Convert the result dictionary to a list of dictionaries
    result_list = [{'location': location, 'status': status} for location, (frequency, status) in result_dict.items()]

    return result_list



def is_status_up_to_date():
    """
    Check if the status is up to date.

    Returns:
        tuple: A tuple containing a boolean value indicating if the status is up to date
               and the old status data if available.
    """
    file_path = "Road/data/deploy/status.json"
    if os.path.exists(file_path) and os.stat(file_path).st_size > 0:
        with open(file_path, "r") as file:
            old_status = json.load(file)

        largest_date = max(old_status, key=lambda x: x['date'])['date']
        largest_date = datetime.datetime.strptime(largest_date, "%Y-%m-%d %H:%M:%S")

        current_datetime = datetime.datetime.now()
        time_difference = current_datetime - largest_date

        return time_difference.total_seconds() < 100, old_status

    return False, None


def read_last_processed_id():
    """
    Read the last processed ID.

    Returns:
        int: The last processed ID.
    """
    with open("Road/data/deploy/last_id.txt", "r+") as file:
        last_id = int(file.read())
    return last_id


def write_last_processed_id(last_id):
    """
    Write the last processed ID to a file.

    Args:
        last_id (int): The last processed ID.
    """
    with open("Road/data/deploy/last_id.txt", "w+") as file:
        file.write(str(last_id))
        
def filter_data(data, last_id):
    """
    Filter the data based on the last processed ID.

    Args:
        data (pd.DataFrame): The data to be filtered.
        last_id (int): The last processed ID.

    Returns:
        pd.DataFrame or None: The filtered data if successful, None otherwise.
    """
    try:
        filtered_data = data[int(last_id) + 1:]
        return filtered_data
    except KeyError:
        print("Invalid start index. Please provide a valid index label.")
    except NameError:
        print("DataFrame 'df' is not defined.")
    return None

def apply_status_pipeline(filtered_data, pipeline):
    """
    Apply the status pipeline on the filtered data.

    Args:
        filtered_data (pd.DataFrame): The filtered data to apply the pipeline on.
        pipeline: The status pipeline to be applied.

    Returns:
        pd.DataFrame: The data with the applied status pipeline.
    """
    # Change the current working directory to a different directory
    current_location = os.getcwd()
    new_location = 'C:/Users/nasser/Desktop/RoadProject/BackEnd/Road/notebooks'
    os.chdir(new_location)
    
    # Apply the status pipeline on the filtered data
    status = pipeline.fit_transform(filtered_data)
    
    # Change the current working directory back to the original directory
    os.chdir(current_location)
    
    return status


def update_status_file(status):
    """
    Update the status in the file.

    Args:
        status (list): The status to be updated in the file.

    Returns:
        list: The updated status.
    """
    file_path = "Road/data/deploy/status.json"
    if os.path.exists(file_path) and os.stat(file_path).st_size > 0:
        with open(file_path, "r+") as file:
            old_status = json.load(file)
    else:
        old_status = []
    
    latest_status = {}
    for item in old_status:
        location = item['location']
        if location not in latest_status or item['date'] > latest_status[location]['date']:
            latest_status[location] = item
    
    for item in status:
        location = item['location']
        if location not in latest_status or item['date'] > latest_status[location]['date']:
            latest_status[location] = item
    
    updated_status = list(latest_status.values())
    
    with open(file_path, "w") as file:
        json.dump(updated_status, file, indent=4)
    return updated_status
