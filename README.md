# RoadProject

In recent years, Israeli barriers have become increasingly prevalent, causing daily hardships for drivers who encounter these barriers on a regular basis. To address this problem, a project has been developed that introduces an application providing real-time data about the barriers in Palestine.
# Files

 - [BackEnd For the Project (Python)](https://github.com/mohamadhase/RoadProject/tree/main/BackEnd)
 - [FrontEnd For the project (Angular)](https://github.com/mohamadhase/RoadProject/tree/main/FrontEnd/map)
 - [Notebooks for models training](https://github.com/mohamadhase/RoadProject/tree/main/BackEnd/Road/notebooks%20models)

## Data

All the data consists of messages collected from a Telegram group called [احوال الطرق وحواجز الاحتلال](https://t.me/Ahwaltareq). The data was collected from February 2023 to May 2023. You can see a snippet of the data [here](https://github.com/mohamadhase/RoadProject/blob/main/BackEnd/Road/data/raw.csv). The system in production uses the same source of data, and the collected data is used to train the models. Real-time data will be used for predictions.

## Process PipeLine
The prediction pipeline consists of four main stages, as shown in the image:

<a href="https://ibb.co/7WMb7hT"><img src="https://i.ibb.co/QNSXs0q/Brainstorming-and-Ideation-1.png" alt="Brainstorming-and-Ideation-1" border="0" /></a>
from the above image we have 4 main stages ion the prediction Pipeline

1.  **Preprocessing Stage**: In this stage, the messages are cleaned and only the necessary columns are retained.
2.  **Data Filtering Stage**: The messages are filtered to keep only the informative ones.
3.  **Information Extraction Stage**: The information is extracted from the messages.
4.  **Output Cleaning Stage**: The extracted information is transformed into pairs of (location, status) for further processing.

## System Design
The system design is illustrated in the following diagram:
<a href="https://ibb.co/7Wdx35R"><img src="https://i.ibb.co/vP6CNGQ/Blank-diagram.jpg" alt="Blank-diagram" border="0"></a>

## Font End 
The front-end demo is developed using Angular, and here are some screenshots of the interface:
<a href="https://ibb.co/Qmq5X7m"><img src="https://i.ibb.co/XJdwVfJ/Screenshot-2023-06-09-192303.png" alt="Screenshot-2023-06-09-192303" border="0"></a><br /><br />

<a href="https://ibb.co/0YKY0sr"><img src="https://i.ibb.co/qmCmHJp/Screenshot-2023-06-09-192251.png" alt="Screenshot-2023-06-09-192251" border="0"></a>
<a href="https://ibb.co/kQRLVyh"><img src="https://i.ibb.co/xJpnKsM/Screenshot-2023-06-09-192244.png" alt="Screenshot-2023-06-09-192244" border="0"></a>
