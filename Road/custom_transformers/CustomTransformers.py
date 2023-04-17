from transformers import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import sys
import os
from utils.helpers import combine_sub_words, word2features
path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if path not in sys.path:
    sys.path.append(path)
    
from utils.clean_data import clean_text
from constants import COLUMNS_WE_NEED
from utils.transform_data import transform_reply_to
import tensorflow as tf
from keras.models import load_model
import numpy as np
import joblib
from transformers import AutoTokenizer, AutoModelForTokenClassification

def print_current_step(step_name):
    print(f"Current step: {step_name}")

class Cleaner(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self
    
    def transform(self, X):
        # cll the print_current_step function with the name of the class as the argument
        print_current_step(self.__class__.__name__)
        # filter columns
        X = X[COLUMNS_WE_NEED]
        # remove rows with no message
        X = X[X['message'].notna()]
        X['message'] = X['message'].apply(clean_text)
        return X
    
class ReplyFinder(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self
    
    def transform(self, X):
        print_current_step(self.__class__.__name__)
        X['reply_to'] = X.apply(transform_reply_to,args=(X,),axis=1)
        X.drop('id', axis=1, inplace=True)
        return X


class QuestionClassifier(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self
    
    def transform(self, X):
        print_current_step(self.__class__.__name__)
        # load the model called question_classifier.h5 in model folder using joblib
        question_classifier =load_model('../models/question_classifier.h5')
        # load the tokenizer called question_tokenizer.joblib in model folder using joblib
        question_tokenizer = joblib.load('../models/question_tokenizer.joblib')
        new_sequences_message = question_tokenizer.texts_to_sequences(X['message'].values)
        new_sequences_reply = question_tokenizer.texts_to_sequences(X['reply_to'].values)
        new_X_message = tf.keras.preprocessing.sequence.pad_sequences(new_sequences_message, maxlen=100)
        new_X_reply = tf.keras.preprocessing.sequence.pad_sequences(new_sequences_reply, maxlen=100)
        # add the predicted class to the dataframe using the index of the large value in the prediction
        message_prediction = question_classifier.predict(new_X_message)
        reply_prediction = question_classifier.predict(new_X_reply)
        X['message_is_question'] = np.argmax(message_prediction, axis=1)
        X['reply_is_question'] = np.argmax(reply_prediction, axis=1)
        return X    
    
class InformationClassifier(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self
    
    def transform(self, X):
        print_current_step(self.__class__.__name__)
        # load the model
        information_classifier = load_model('../models/information_classifier.h5')
        # load the tokenizer
        information_tokenizer = joblib.load('..\models\information_tokenizer.joblib')
        # Convert the text data to sequences of integers
        new_sequences = information_tokenizer.texts_to_sequences(X["message"])
        new_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(new_sequences, maxlen=20, padding="post", truncating="post")
        # do the same for the reply_to column
        new_sequences2 = information_tokenizer.texts_to_sequences(X["reply_to"])
        new_sequences_padded2 = tf.keras.preprocessing.sequence.pad_sequences(new_sequences2, maxlen=20, padding="post", truncating="post")
        # predict the output
        predictions = (information_classifier.predict([new_sequences_padded, new_sequences_padded2, np.array(X[["message_is_question", "reply_is_question"]])]))
        X['is_giving_information'] = predictions[:,0]
        # convert the output to a binary value
        X['is_giving_information'] = X['is_giving_information'].apply(lambda x: 1 if x > 0.5 else 0)
        # drop the is_message_question and is_reply_question columns
        X.drop(['message_is_question', 'reply_is_question'], axis=1, inplace=True)
        return X
    
    
class CombineMessageReply(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self
    
    def transform(self, X):
        print_current_step(self.__class__.__name__)
        X['full_text'] = X['reply_to'] + ' ' + X['message']
        X.drop(['message','reply_to'], axis=1, inplace=True)
        # make the full_text column the first column
        cols = X.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        X = X[cols]
        return X


# class HajezNameClassifier(BaseEstimator, TransformerMixin):
    
#     def __init__(self):
#         pass
#     def fit(self, X, y=None):
#         # Do any necessary preprocessing that requires access to the training data
#         # and save any parameters that will be used in the transform method
#         return self
    
#     def transform(self, X):
#         print_current_step(self.__class__.__name__)
#         # load the model
#         hajez_classifier = load_model('../models/hajez_classifier.h5')
#         # load the tokenizer
#         hajez_tokenizer = joblib.load('../models/hajez_tokenizer.pkl')
#         # convert text data to sequences
#         sequences = hajez_tokenizer.texts_to_sequences(X["full_text"])
#         padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=205)
#         binary_features = X[['message_is_question', 'reply_is_question', 'is_giving_information']].values
#         # Predict the categories
#         predictions = hajez_classifier.predict([padded_sequences, binary_features])
#         predicted_categories = [INDEX_TO_HAJEZ[np.argmax(prediction)] for prediction in predictions]
#         X["hajez_name"] = predicted_categories
#         return X
    
class NerClassifier(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self
    
    def transform(self, X):
        print(self.__class__.__name__)
        print("hi")
        # load the model
        model = load_model('../models/ner/ner_model.h5')
        # load the tokenizer
        input_tokenizer = joblib.load('../models/ner/input_tokenizer.pkl')
        #load dictionary
        output_idx2word = joblib.load('../models/ner/output_idx2word.pkl')
        
        new_input_sequence = input_tokenizer.texts_to_sequences(X['full_text'].values)
        new_input_sequence = tf.keras.preprocessing.sequence.pad_sequences(new_input_sequence, maxlen=205, padding='post')
        pred_output_seq = model.predict(new_input_sequence)
        pred_output_seq = np.argmax(pred_output_seq, axis=-1)
        pred_named_entities = [[output_idx2word.get(idx, '') for idx in seq] for seq in pred_output_seq]
        ner_str = []
        # map each word to its predicted named entity
        for text, pred_named_entity in zip(X['full_text'].values, pred_named_entities):
            ner_str.append([f"{ner} " if ner != '' else "O " for word, ner in zip(text.split(), pred_named_entity)])
        X['ner'] = ner_str
        return X
    
    
    
    
# class NerCrfClassifier(BaseEstimator, TransformerMixin):


#     def __init__(self):
#         self.model_name = "CAMeL-Lab/bert-base-arabic-camelbert-ca-pos-egy"
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.pos_pipeline = pipeline("ner", model=self.model_name, tokenizer=self.tokenizer)
    
#     def fit(self, X, y=None):
#         # Do any necessary preprocessing that requires access to the training data
#         # and save any parameters that will be used in the transform method
#         return self
#     def sent2features(self,sent):
#         encoded_input = self.pos_pipeline(sent)
#         encoded_input = combine_sub_words(encoded_input)
#         pos = [word["entity"] for word in encoded_input]
#         sent = sent.split()
#         res = [word2features(sent,pos, i) for i in range(len(sent))]

#         return res

    
#     def transform(self, X):
#         print_current_step(self.__class__.__name__)
#         text_to_fetures = []
#         for text in X['full_text']:
#             text_to_fetures.append(self.sent2features(text))
#         # load the model from disk
#         loaded_model = joblib.load('../models/NER_CRF.pkl')
#         # predict the result
#         y_pred = loaded_model.predict(text_to_fetures)
#         X['ner'] = y_pred
#         return X
    
    
    
class GetLocationList(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self
    def transform(self, X):
        print_current_step(self.__class__.__name__)
        # create new column contains the words that has ner tag B-LOC
        locs =[]
        for index, row in X.iterrows():
            text = row['full_text'].split()
            ner_tags = row['ner']
            loc_words = []
            word = ''
            for i in range(len(text)-1,-1,-1):
                if ner_tags[i] == 'I-LOC ':
                    word = f'{text[i]} '
                if ner_tags[i] == 'B-LOC ':
                    word = f'{text[i]} {word}'
                    loc_words.append(word)
                    word = ''
            locs.append(loc_words)

        X['locs'] = locs
        # remove the duplication in each locs list
        X['locs'] = X['locs'].apply(lambda x: list(set(x)))
        return X
    
class GetStatusList(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self
    def transform(self, X):
        print_current_step(self.__class__.__name__)
        # create new column contains the words that has ner tag B-LOC
        locs =[]
        for index, row in X.iterrows():
            text = row['full_text'].split()
            ner_tags = row['ner']
            loc_words = []

            for i in range(len(text)):
                if ner_tags[i] == 'STAT ':
                    word = f'{text[i]} '
                    loc_words.append(word)
            locs.append(loc_words)

        X['Stat'] = locs
        # remove the duplication in each locs list
        X['Stat'] = X['Stat'].apply(lambda x: list(set(x)))
        return X
    
    
class FilterIsGivingInformation(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self
    def transform(self, X):
        print_current_step(self.__class__.__name__)
        # remove any row has is_giving_information = 0
        X = X[X['is_giving_information'] == 1]
        # drop the is_giving_information column
        X = X.drop(columns=['is_giving_information'])
        return X
    
    
    
class FilterHasLoc(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self
    def transform(self, X):
        print_current_step(self.__class__.__name__)
        # remove any row has loc = []
        X = X[X['locs'].apply(lambda x: len(x) > 0)]

        return X
    
    
class FilterHasStat(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self
    def transform(self, X):
        print_current_step(self.__class__.__name__)
        # remove any row has stat = []
        X = X[X['Stat'].apply(lambda x: len(x) > 0)]
        return X
    
    
class GenerateSummary(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Do any necessary preprocessing that requires access to the training data
        # and save any parameters that will be used in the transform method
        return self
    def transform(self, X):
        # TODO: Handle the case where there are more than one loc or stat
        
        
        print_current_step(self.__class__.__name__)
        # keep only the rows with one loc and one stat
        print(X.shape)
        X = X[X['locs'].apply(lambda x: len(x) == 1)]
        print(X.shape)
        X = X[X['Stat'].apply(lambda x: len(x) == 1)]
        print(X.shape)
        # generate the summary
        X['summary'] = X.apply(lambda x: {
            "location": x['locs'][0],
            "status": x['Stat'][0],
            }, axis=1)
        
        summary_array = X['summary'].to_numpy()
        
        return summary_array