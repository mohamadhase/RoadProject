from http import HTTPStatus
from fastapi import Request
from pydantic import BaseModel, Field
from BackEnd.Road.utils.helpers import (apply_status_pipeline,
                                        filter_data, is_status_up_to_date, read_last_processed_id,
                                        update_status_file, write_last_processed_id)
from Road import app, logger
from Road.constants import HAWAJEZ_LONG_LAT
from Road.utils.helpers import group_locations
import pandas as pd
import csv
from Road import pipeline
from collections import Counter
import datetime

class Message(BaseModel):
    """
    Represents a message.

    Attributes:
        id (int): The ID of the message.
        message (str): The content of the message.
        reply_to (dict, optional): The dictionary representing the reply details.
            Defaults to None.
    """
    id: int
    message: str
    reply_to: dict = Field(default=None, title="Reply To")

@app.get("/health")
def _health_check(request: Request) -> dict:
    """
    Endpoint for health check.
    
    Args:
        request (Request): The FastAPI request object.
        
    Returns:
        dict: A dictionary containing the health check response.
    """
    logger.info("Performing health check.")
    
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }

    logger.info(f"Health check status code: {HTTPStatus.OK}.")
    return response

from fastapi import Request
from http import HTTPStatus

@app.get("/hawajez_points")
def __get_hawajez_points(request: Request) -> dict:
    """
    Endpoint for retrieving Hawajez Points.
    
    Args:
        request (Request): The FastAPI request object.
        
    Returns:
        dict: A dictionary containing the Hawajez Points response.
    """
    logger.info("Retrieving Hawajez Points.")
    
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": HAWAJEZ_LONG_LAT,
    }
    
    logger.info(f"Hawajez Points status code: {HTTPStatus.OK}.")
    return response




@app.post("/message")
async def post_message(data: Message) -> dict:
    """
    Endpoint for posting a message.

    Args:
        data (Message): The message data to be posted.

    Returns:
        dict: A dictionary indicating a successful post.
    """
    data = data.dict()
    
    # Append the data to a CSV file called publish_data.csv
    with open("Road/data/deploy/data.csv", "a+", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())

        # Check if the file is empty (to write the header only once)
        if file.tell() == 0:
            writer.writeheader()
    
        writer.writerow(data)

    logger.info(f"Message posted successfully. ID: {data['id']}")
    logger.debug(f"Posted message content: {data['message']}")
    
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": "Message posted successfully.",
    }


@app.get("/hawajez_status")
def __get_hawajez_status(request: Request) -> dict:
    """
    Endpoint for retrieving Hawajez status.

    Args:
        request (Request): The FastAPI request object.

    Returns:
        dict: A dictionary containing the Hawajez status response.
    """
    logger.info("Retrieving Hawajez status.")

    # Check if the status is up to date
    is_up_to_date, old_status = is_status_up_to_date()
    print(is_up_to_date, old_status)
    if is_up_to_date:
        response = {
            "message": "Status is up to date",
            "status-code": HTTPStatus.OK,
            "data": old_status
        }
        logger.info(f"Hawajez status is up to date. Status-code: {HTTPStatus.OK}")
        return response

    # Read the last processed ID
    last_id = read_last_processed_id()

    # Read the data from the CSV file
    data = pd.read_csv("Road/data/deploy/data.csv")

    # Write the last processed ID
    write_last_processed_id(data.index[-1])

    # Filter the data
    filtered_data = filter_data(data, last_id)

    if filtered_data is None:
        response = {
            "message": "Error filtering data",
            "status-code": HTTPStatus.INTERNAL_SERVER_ERROR,
            "data": None
        }
        logger.error(f"Error filtering data. Status-code: {HTTPStatus.INTERNAL_SERVER_ERROR}")
        return response

    # Apply the status pipeline
    status = apply_status_pipeline(filtered_data, pipeline)

    # Group the locations
    status = group_locations(status)

    # Add the current date and time to each status item
    status = [{"date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **item} for item in status]

    # Update the status in the file
    updated_status = update_status_file(status)

    # Prepare the response
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": updated_status,
    }
    logger.info(f"Hawajez status retrieved successfully. Status-code: {HTTPStatus.OK}")
    return response