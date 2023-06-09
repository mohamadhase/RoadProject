import pandas as pd
import ast
import os
import sys
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(os.path.abspath(parent_path))
from utils.helpers import is_question_arabic, is_it_talk_about_hajez, is_it_give_information

def transform_reply_to(row: pd.Series, data: pd.DataFrame) -> str:
    """
    Extracts the text of the replied message, given a row from a DataFrame.

    Args:
        row (pandas.Series): A row from a DataFrame representing a message.
        data (pandas.DataFrame): A DataFrame containing all the messages.

    Returns:
        str: The text of the message that was replied to, or an empty string if no message was replied to.
    """
    reply_text = ""

    # Check if the 'reply_to' field is empty
    if pd.isna(row["reply_to"]):
        return reply_text

    # Parse the 'reply_to' field as a dictionary
    reply_to = ast.literal_eval(row["reply_to"])

    # Extract the message ID of the replied message
    reply_to_id = reply_to["reply_to_msg_id"]

    # Get the replied message from the DataFrame by matching the ID using loc instead of boolean indexing
    data_reply_to = data.loc[data["id"] == reply_to_id, "message"]

    # If a message was found, return its text as the 'reply_text'
    if not data_reply_to.empty:
        reply_text = data_reply_to.iat[0]

    return reply_text

def combine_message_and_reply(row: pd.Series) -> str:
    """
    Combines the message and reply into a single string, given a row from a DataFrame.

    Args:
        row (pandas.Series): A row from a DataFrame representing a message.

    Returns:
        str: The combined message and reply, or an empty string if the reply is empty or the message type is not supported.
    """
    if row["reply_clean_text"] == "":
        if row['message_type'] == "question":
            return ""
        elif is_it_talk_about_hajez(row["message_clean_text"]) and is_it_give_information(row["message_clean_text"]):
            return row["message_clean_text"]
        else:
            return ""
    if is_it_talk_about_hajez(row["reply_clean_text"]) and is_it_give_information(row["message_clean_text"]) and row['message_type'] == "statement":
        return row["reply_clean_text"] + "|||" + row["message_clean_text"]
    else:
        return ""