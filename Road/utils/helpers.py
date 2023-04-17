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


def is_question_arabic(text):
    """
    Check if the given text is a question in Arabic.

    Args:
        text (str): The text to check.

    Returns:
        bool: True if the text is a question in Arabic, False otherwise.
    """
    return any(fuzz.partial_ratio(word, text) >= 80 for word in QUESTION_WORDS)


# def draw_word_cloud(all_text, file_name):
#     """
#     Draw a word cloud from the given text and save it to a file.

#     Args:
#         all_text (str): The text to generate the word cloud from.
#         file_name (str): The name of the file to save the word cloud to.
#     """
#     all_text = re.sub(r'[^\u0600-\u06FF\s]+', '', all_text)
#     reshaped_text = arabic_reshaper.reshape(all_text)
#     display_text = get_display(reshaped_text)
#     wordcloud = WordCloud(width=800, height=800, font_path='C:/Windows/Fonts/Arial.ttf', background_color='white').generate(display_text)

#     plt.figure(figsize=(8, 8), facecolor=None)
#     plt.imshow(wordcloud)
#     plt.axis('off')
#     plt.tight_layout(pad=0)
#     # save the wordcloud in high resolution to a file
#     plt.savefig(f'../data/{file_name}.png', dpi=300)


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
    Returns True if the word is punctuation, False otherwise.
    """
    # check if the word is punctuation
    punct_chars = set('؟،؛?!.,:;')
    return all(char in punct_chars for char in word)



def combine_sub_words(output):
    for index in range(len(output)):
        if index < len(output) - 1:
            if '##' in output[index+1]['word']:
                output[index]['word'] = output[index]['word'] + output[index+1]['word']
                output[index]['word'] = output[index]['word'].replace('##', '')
                # delete the next word
                output.pop(index+1)
    return output


def word2features(sent,pos, i):
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
            # 'prev_word_barr': is_it_bar(prev_word),
            # 'prev_word_stat': is_it_status(prev_word),
            'prev_word_pos': pos[i-1],
        })
    if i < len(sent)-1:
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
            # 'next_word_barr': is_it_bar(next_word),
            # 'next_word_stat': is_it_status(next_word),
            'next_word_pos': pos[i+1]

        })
    return features

  
