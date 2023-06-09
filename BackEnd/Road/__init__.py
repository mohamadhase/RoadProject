from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import logging
import os
import sys
from telethon import TelegramClient
import csv
from telethon import events
import asyncio

# Get the parent directory path
path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
# Define paths for the backend and notebooks directories
path2 = path + '\BackEnd\Road'
path3 = path + '\BackEnd\Road\\notebooks'
# Add the paths to the sys.path to import modules from these directories
if path not in sys.path:
    sys.path.append(path)
    sys.path.append(path2)
    sys.path.append(path3)

# Create a FastAPI instance
app = FastAPI(
    title="Road Status API",
    description="",
    version=0.1
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained pipeline
pipeline = joblib.load('Road/models/pipeline/pipeline.pkl')

# Configure the logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename='Road/logs/Road.log', level=logging.DEBUG)

# Import endpoints from the Road module
from Road.endpoints import endpoints
