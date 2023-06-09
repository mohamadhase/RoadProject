from sklearn.base import BaseEstimator, TransformerMixin
import sys
import os
path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if path not in sys.path:
    sys.path.append(path)
from constants import STATUS_MAPPER, STATUS_NAMES

from utils.clean_data import clean_text
from constants import COLUMNS_WE_NEED,HAWAJEZ_NAMES
from utils.transform_data import transform_reply_to
import tensorflow as tf
from keras.models import load_model
import numpy as np
import joblib
import jellyfish



def print_current_step(step_name):
    """
    Prints the current step name.
    """
    print(f"Current step: {step_name}")


class Cleaner(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for data cleaning.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer to the training data.

        Parameters:
        - X: Input data (DataFrame or array-like)
        - y: Target labels (optional)

        Returns:
        - self: The fitted transformer object.
        """
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self

    def transform(self, X):
        """
        Transforms the input data by performing cleaning operations.

        Parameters:
        - X: Input data (DataFrame or array-like)

        Returns:
        - X: Cleaned data (DataFrame or array-like)
        """
        # Check if X is empty and return empty X if true
        if X.empty:
            return X

        # Call the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)

        # Filter columns
        X = X[COLUMNS_WE_NEED]

        # Remove rows with no message
        X = X[X['message'].notna()]

        # Apply text cleaning function to the 'message' column
        X['message'] = X['message'].apply(clean_text)

        return X
    
class ReplyFinder(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for finding replies in data.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer to the training data.

        Parameters:
        - X: Input data (DataFrame or array-like)
        - y: Target labels (optional)

        Returns:
        - self: The fitted transformer object.
        """
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self

    def transform(self, X):
        """
        Transforms the input data by finding replies.

        Parameters:
        - X: Input data (DataFrame or array-like)

        Returns:
        - X: Transformed data (DataFrame or array-like)
        """
        # Call the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)

        # Check if X is empty and return empty X if true
        if X.empty:
            return X

        # Apply transform_reply_to function to create 'reply_to' column
        X['reply_to'] = X.apply(transform_reply_to, args=(X,), axis=1)

        # Drop 'id' column
        X.drop('id', axis=1, inplace=True)

        return X


class QuestionClassifier(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for classifying questions in data.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer to the training data.

        Parameters:
        - X: Input data (DataFrame or array-like)
        - y: Target labels (optional)

        Returns:
        - self: The fitted transformer object.
        """
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self

    def transform(self, X):
        """
        Transforms the input data by classifying questions.

        Parameters:
        - X: Input data (DataFrame or array-like)

        Returns:
        - X: Transformed data (DataFrame or array-like)
        """
        # Call the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)

        # Check if X is empty and return empty X if true
        if X.empty:
            return X

        # Load the question_classifier model
        question_classifier = load_model('../models/question_classifier.h5')

        # Load the question_tokenizer
        question_tokenizer = joblib.load('../models/question_tokenizer.joblib')

        # Convert 'message' and 'reply_to' columns to sequences
        new_sequences_message = question_tokenizer.texts_to_sequences(X['message'].values)
        new_sequences_reply = question_tokenizer.texts_to_sequences(X['reply_to'].values)

        # Pad the sequences to a fixed length of 100
        new_X_message = tf.keras.preprocessing.sequence.pad_sequences(new_sequences_message, maxlen=100)
        new_X_reply = tf.keras.preprocessing.sequence.pad_sequences(new_sequences_reply, maxlen=100)

        # Predict the class for 'message' and 'reply_to'
        message_prediction = question_classifier.predict(new_X_message)
        reply_prediction = question_classifier.predict(new_X_reply)

        # Add the predicted class to the dataframe using the index of the largest value in the prediction
        X['message_is_question'] = np.argmax(message_prediction, axis=1)
        X['reply_is_question'] = np.argmax(reply_prediction, axis=1)

        return X
 
    
class InformationClassifier(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for classifying information in data.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer to the training data.

        Parameters:
        - X: Input data (DataFrame or array-like)
        - y: Target labels (optional)

        Returns:
        - self: The fitted transformer object.
        """
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self

    def transform(self, X):
        """
        Transforms the input data by classifying information.

        Parameters:
        - X: Input data (DataFrame or array-like)

        Returns:
        - X: Transformed data (DataFrame or array-like)
        """
        # Call the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)

        # Check if X is empty and return empty X if true
        if X.empty:
            return X

        # Load the information_classifier model
        information_classifier = load_model('../models/information_classifier.h5')

        # Load the information_tokenizer
        information_tokenizer = joblib.load('../models/information_tokenizer.joblib')

        # Convert 'message' and 'reply_to' columns to sequences
        new_sequences = information_tokenizer.texts_to_sequences(X["message"])
        new_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=20, padding="post", truncating="post")

        new_sequences2 = information_tokenizer.texts_to_sequences(X["reply_to"])
        new_sequences_padded2 = tf.keras.preprocessing.sequence.pad_sequences(new_sequences2, maxlen=20, padding="post", truncating="post")

        # Predict the output
        predictions = information_classifier.predict([new_sequences_padded, new_sequences_padded2, np.array(X[["message_is_question", "reply_is_question"]])])

        # Add the predicted class to the dataframe
        X['is_giving_information'] = predictions[:, 0]

        # Convert the output to a binary value
        X['is_giving_information'] = X['is_giving_information'].apply(lambda x: 1 if x > 0.5 else 0)

        # Drop the 'message_is_question' and 'reply_is_question' columns
        X.drop(['message_is_question', 'reply_is_question'], axis=1, inplace=True)

        return X

    
    
class CombineMessageReply(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for combining 'message' and 'reply_to' columns into a single 'full_text' column.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer to the training data.

        Parameters:
        - X: Input data (DataFrame or array-like)
        - y: Target labels (optional)

        Returns:
        - self: The fitted transformer object.
        """
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self

    def transform(self, X):
        """
        Transforms the input data by combining 'message' and 'reply_to' columns into a single 'full_text' column.

        Parameters:
        - X: Input data (DataFrame or array-like)

        Returns:
        - X: Transformed data (DataFrame or array-like)
        """
        # Call the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)

        # Check if X is empty and return empty X if true
        if X.empty:
            return X

        # Combine 'reply_to' and 'message' columns into 'full_text'
        X['full_text'] = X['reply_to'] + ' ' + X['message']

        # Drop 'message' and 'reply_to' columns
        X.drop(['message', 'reply_to'], axis=1, inplace=True)

        # Make the 'full_text' column the first column
        cols = X.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        X = X[cols]

        return X
    
    
class NerClassifier(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for classifying named entities in text data.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer to the training data.

        Parameters:
        - X: Input data (DataFrame or array-like)
        - y: Target labels (optional)

        Returns:
        - self: The fitted transformer object.
        """
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self

    def transform(self, X):
        """
        Transforms the input data by classifying named entities in text data.

        Parameters:
        - X: Input data (DataFrame or array-like)

        Returns:
        - X: Transformed data (DataFrame or array-like)
        """
        # Call the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)

        # Check if X is empty and return empty X if true
        if X.empty:
            return X

        # Load the NER model
        model = load_model('../models/ner/ner_model.h5')

        # Load the input tokenizer
        input_tokenizer = joblib.load('../models/ner/input_tokenizer.pkl')

        # Load the output idx2word dictionary
        output_idx2word = joblib.load('../models/ner/output_idx2word.pkl')

        # Convert 'full_text' column to input sequences
        new_input_sequence = input_tokenizer.texts_to_sequences(X['full_text'].values)
        new_input_sequence = tf.keras.preprocessing.sequence.pad_sequences(new_input_sequence, maxlen=205, padding='post')

        # Predict the output sequences
        pred_output_seq = model.predict(new_input_sequence)
        pred_output_seq = np.argmax(pred_output_seq, axis=-1)

        # Map the predicted output indices to named entities
        pred_named_entities = [[output_idx2word.get(idx, '') for idx in seq] for seq in pred_output_seq]

        ner_str = []
        # Map each word to its predicted named entity
        for text, pred_named_entity in zip(X['full_text'].values, pred_named_entities):
            ner_str.append([f"{ner} " if ner != '' else "O " for word, ner in zip(text.split(), pred_named_entity)])

        # Add the 'ner' column to the dataframe
        X['ner'] = ner_str

        return X

    
class GetLocationList(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for extracting location lists from text data.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer to the training data.

        Parameters:
        - X: Input data (DataFrame or array-like)
        - y: Target labels (optional)

        Returns:
        - self: The fitted transformer object.
        """
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self

    def transform(self, X):
        """
        Transforms the input data by extracting location lists from text data.

        Parameters:
        - X: Input data (DataFrame or array-like)

        Returns:
        - X: Transformed data (DataFrame or array-like)
        """
        # Call the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)

        # Check if X is empty and return empty X if true
        if X.empty:
            return X

        # Create a new column to store the extracted location lists
        locs = []

        # Iterate over each row in the dataframe
        for index, row in X.iterrows():
            text = row['full_text'].split()
            ner_tags = row['ner']
            loc_words = []
            word = ''
            # Iterate over each word and its corresponding NER tag in reverse order
            for i in range(len(text)-1, -1, -1):
                if ner_tags[i] == 'I-LOC ':
                    word = f'{text[i]} '
                if ner_tags[i] == 'B-LOC ':
                    word = f'{text[i]} {word}'
                    loc_words.append(word)
                    word = ''
            locs.append(loc_words)

        # Assign the extracted location lists to the 'locs' column
        X['locs'] = locs

        # Remove duplicate locations in each 'locs' list
        X['locs'] = X['locs'].apply(lambda x: list(set(x)))

        return X

    
class GetStatusList(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for extracting status lists from text data.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer to the training data.

        Parameters:
        - X: Input data (DataFrame or array-like)
        - y: Target labels (optional)

        Returns:
        - self: The fitted transformer object.
        """
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self

    def transform(self, X):
        """
        Transforms the input data by extracting status lists from text data.

        Parameters:
        - X: Input data (DataFrame or array-like)

        Returns:
        - X: Transformed data (DataFrame or array-like)
        """
        # Call the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)

        # Check if X is empty and return empty X if true
        if X.empty:
            return X

        # Create a new column to store the extracted status lists
        statuses = []

        # Iterate over each row in the dataframe
        for index, row in X.iterrows():
            text = row['full_text'].split()
            ner_tags = row['ner']
            status_words = []

            # Iterate over each word and its corresponding NER tag
            for i in range(len(text)):
                if ner_tags[i] == 'STAT ':
                    word = f'{text[i]} '
                    status_words.append(word)

            statuses.append(status_words)

        # Assign the extracted status lists to the 'Stat' column
        X['Stat'] = statuses

        # Remove duplicate statuses in each 'Stat' list
        X['Stat'] = X['Stat'].apply(lambda x: list(set(x)))

        # Drop the 'ner' column
        X = X.drop(columns=['ner'])

        return X

    
    
class FilterIsGivingInformation(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for filtering rows based on the 'is_giving_information' column.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer to the training data.

        Parameters:
        - X: Input data (DataFrame or array-like)
        - y: Target labels (optional)

        Returns:
        - self: The fitted transformer object.
        """
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self

    def transform(self, X):
        """
        Transforms the input data by filtering rows based on the 'is_giving_information' column.

        Parameters:
        - X: Input data (DataFrame or array-like)

        Returns:
        - X: Transformed data (DataFrame or array-like)
        """
        # Call the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)

        # Check if X is empty and return empty X if true
        if X.empty:
            return X

        # Filter rows where 'is_giving_information' is 1
        X = X[X['is_giving_information'] == 1]

        return X.drop(columns=['is_giving_information'])

    
    
class FilterHasLoc(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for filtering rows based on the presence of location information.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer to the training data.

        Parameters:
        - X: Input data (DataFrame or array-like)
        - y: Target labels (optional)

        Returns:
        - self: The fitted transformer object.
        """
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self

    def standarize_loc(self, locs):
        """
        Standarizes the location values based on a predefined list of similar names.

        Parameters:
        - locs: List of location values

        Returns:
        - new_locs: List of standardized location values
        """
        new_locs = []
        for i in range(len(locs)):
            loc = locs[i]
            loc = loc.strip()

            # Find the most similar word in HAWAJEZ_NAMES and replace it if the similarity is greater than 0.8
            max_sim = 0
            max_index = -1

            for index, name in enumerate(HAWAJEZ_NAMES):
                sim = jellyfish.jaro_winkler(loc, name)
                if sim > max_sim and sim > 0.8:
                    max_sim = sim
                    max_index = index
                    break

            if max_index != -1:
                new_locs.append(HAWAJEZ_NAMES[max_index])

        return list(set(new_locs))

    def transform(self, X):
        """
        Transforms the input data by filtering rows based on the presence of location information.

        Parameters:
        - X: Input data (DataFrame or array-like)

        Returns:
        - X: Transformed data (DataFrame or array-like)
        """
        # Call the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)

        # Check if X is empty and return empty X if true
        if X.empty:
            return X

        # Apply the standarize_loc method to the 'locs' column
        X['locs'] = X['locs'].apply(self.standarize_loc)

        # Remove rows with empty location lists
        X = X[X['locs'].apply(lambda x: len(x) > 0)]

        return X

    
    
class FilterHasStat(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for filtering rows based on the presence of status information.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer to the training data.

        Parameters:
        - X: Input data (DataFrame or array-like)
        - y: Target labels (optional)

        Returns:
        - self: The fitted transformer object.
        """
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self

    def standarize_Stat(self, stats):
        """
        Standardizes the status values based on a predefined list of similar names.

        Parameters:
        - stats: List of status values

        Returns:
        - new_stats: List of standardized status values
        """
        new_stats = []
        for i in range(len(stats)):
            stat = stats[i]
            stat = stat.strip()

            # Find the most similar word in STATUS_NAMES and replace it if the similarity is greater than 0.8
            max_sim = 0
            max_index = -1

            for index, name in enumerate(STATUS_NAMES):
                sim = jellyfish.jaro_winkler(stat, name)
                if sim > max_sim and sim > 0.8:
                    max_sim = sim
                    max_index = index
                    break

            if max_index != -1:
                new_stats.append(STATUS_NAMES[max_index])

        return list(set(new_stats))

    def map_stat(self, stats):
        """
        Maps status values to their corresponding mapping values.

        Parameters:
        - stats: List of status values

        Returns:
        - new_stats: List of mapped status values
        """
        new_stats = []
        for stat in stats:
            for key in STATUS_MAPPER.keys():
                if stat in key:
                    new_stats.append(STATUS_MAPPER[key])
                    break
            else:
                new_stats.append(stat)

        return list(set(new_stats))

    def solve_conflit(self, stats):
        """
        Solves conflicts between different status values.

        Parameters:
        - stats: List of status values

        Returns:
        - new_stats: List of resolved status values
        """
        # If status has 'مفتوح' and 'مسكر', keep the last one
        if len(stats) > 1 and ('مفتوح' in stats and 'مسكر' in stats):
            return [stats[-1]]

        return stats

    def transform(self, X):
        """
        Transforms the input data by filtering rows based on the presence of status information.

        Parameters:
        - X: Input data (DataFrame or array-like)

        Returns:
        - X: Transformed data (DataFrame or array-like)
        """
        # Call the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)

        # Check if X is empty and return empty X if true
        if X.empty:
            return X

        # Apply the standarize_Stat method to the 'Stat' column
        X['Stat'] = X['Stat'].apply(self.standarize_Stat)

        # Remove rows with empty status lists
        X = X[X['Stat'].apply(lambda x: len(x) > 0)]

        # Apply the map_stat method to the 'Stat' column
        X['Stat'] = X['Stat'].apply(self.map_stat)

        # Apply the solve_conflit method to the 'Stat' column
        X['Stat'] = X['Stat'].apply(self.solve_conflit)

        return X

    
class GenerateSummary(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for generating summaries based on location and status information.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer to the training data.

        Parameters:
        - X: Input data (DataFrame or array-like)
        - y: Target labels (optional)

        Returns:
        - self: The fitted transformer object.
        """
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self

    def get_summary(self, row):
        """
        Generates summaries based on the row's location and status information.

        Parameters:
        - row: Input row from the DataFrame

        Returns:
        - summary: List of summary dictionaries
        """
        summary = []

        # If the row has 1 location and 1 status, return one summary
        if len(row['locs']) == 1 and len(row['Stat']) == 1:
            summary.append({
                "location": row['locs'][0],
                "status": row['Stat'][0],
            })
        # If the row has 2 locations and 1 status, return two summaries
        elif len(row['locs']) == 2 and len(row['Stat']) == 1:
            summary.append({
                "location": row['locs'][0],
                "status": row['Stat'][0],
            })
            summary.append({
                "location": row['locs'][1],
                "status": row['Stat'][0],
            })
        # If the row has 1 location and 2 statuses, return two summaries
        elif len(row['locs']) == 1 and len(row['Stat']) == 2:
            summary.append({
                "location": row['locs'][0],
                "status": row['Stat'][0],
            })
            summary.append({
                "location": row['locs'][0],
                "status": row['Stat'][1],
            })
        # If the row has 2 locations and 2 statuses, return four summaries
        elif len(row['locs']) == 2 and len(row['Stat']) == 2:
            summary.append({
                "location": row['locs'][0],
                "status": row['Stat'][0],
            })
            summary.append({
                "location": row['locs'][1],
                "status": row['Stat'][1],
            })
        # If the row has 3 locations and 1 status, return three summaries
        elif len(row['locs']) == 3 and len(row['Stat']) == 1:
            summary.append({
                "location": row['locs'][0],
                "status": row['Stat'][0],
            })
            summary.append({
                "location": row['locs'][1],
                "status": row['Stat'][0],
            })
            summary.append({
                "location": row['locs'][2],
                "status": row['Stat'][0],
            })

        return summary

    def transform(self, X):
        """
        Transforms the input data by generating summaries based on location and status information.

        Parameters:
        - X: Input data (DataFrame or array-like)

        Returns:
        - concatenated_list: List of summary dictionaries
        """
        # Call the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)

        # Check if X is empty and return an empty list if true
        if X.empty:
            return []

        # Apply the get_summary method to each row and store the results in the 'summary' column
        X['summary'] = X.apply(self.get_summary, axis=1)

        return [item for sublist in X['summary'].to_numpy() for item in sublist]

    
class FilterLen(BaseEstimator, TransformerMixin):
    """
    Custom transformer class for filtering rows based on the lengths of status and loc columns.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fits the transformer to the training data.

        Parameters:
        - X: Input data (DataFrame or array-like)
        - y: Target labels (optional)

        Returns:
        - self: The fitted transformer object.
        """
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self

    def transform(self, X):
        """
        Transforms the input data by filtering rows based on the lengths of status and loc columns.

        Parameters:
        - X: Input data (DataFrame or array-like)

        Returns:
        - filtered_X: Filtered DataFrame
        """
        # Call the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)

        # Check if X is empty and return X if true
        if X.empty:
            return X

        # Calculate the lengths of the status and loc columns
        X['status_len'] = X['Stat'].apply(lambda x: len(x))
        X['loc_len'] = X['locs'].apply(lambda x: len(x))

        # Filter the rows based on the length conditions
        filtered_X = X[
            (X['status_len'] <= 1) |
            ((X['status_len'] > 1) & (X['loc_len'] == 1)) |
            ((X['status_len'] > 1) & (X['loc_len'] == 2))
        ]

        return filtered_X[filtered_X['loc_len'] <= 3]
