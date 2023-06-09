import uvicorn
from Road import app, logger

if __name__ == "__main__":
    # Get the file name without the extension
    file_name = __file__.split("/")[-1].split(".")[0].split("\\")[-1]
    
    # Log a message indicating that the server is starting
    logger.info("Server is trying to start...")

    # Run the FastAPI app using uvicorn
    uvicorn.run(f"{file_name}:app", host='0.0.0.0', port=8000, reload=True)

    # Log a message indicating that the server started successfully
    logger.info("Server started successfully...")
