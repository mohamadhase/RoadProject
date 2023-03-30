import uvicorn
from Road import app, logger

if __name__ == "__main__":
    file_name = __file__.split("/")[-1].split(".")[0].split("\\")[-1]
    logger.info("Server is Trying to Start ... ")
    uvicorn.run(f"{file_name}:app", host='0.0.0.0', port=8000, reload=True)
    logger.info("Server is Start Succesfully ... ")

