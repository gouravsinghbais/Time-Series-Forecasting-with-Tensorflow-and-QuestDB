# Time-Series-Forecasting-with-Tensorflow-and-QuestDB

### Install QuestDB:

To install QuestDB on any OS platform you will have to go through the following steps:
1. Install Docker from https://docs.docker.com/engine/install/, as QuestDB would run on the Docker platform.
2. Once the docker is installed you need to pull the QuestDB image and create a docker container. To pull the docker image you need to open your command prompt and write the following command:
    ```
    docker run -p 9000:9000 -p 8812:8812 questdb/questdb
    ```
    here, 9000 is the port on which QuestDB would run, 8812 is for Postgres wire protocol. 
3. Open another terminal and run the following command to check if the QuestDB is running or not.
    ```
    docker ps
    ```
    alternatively, you can browse **localhost:9000** and QuestDB should be accessible there. 
    
### Install Tensorflow:
```
pip install tensorflow
```
Note that TensorFlow 1.15 is compatible with python 3.6 and for versions after TensorFlow 2.0 is compatible so you need to make sure you are using python 3.6. 

To check the QuestDB operations you can visit the following [link](https://tutswiki.com/setup-access-questdb-python-notebook/).
