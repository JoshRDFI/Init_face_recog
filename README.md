# Init_face_recog
My first foray into facial recognition

This project is based off Matthew Berman's video (and Google Colab notebook) found here: [Youtube Video](https://www.youtube.com/watch?v=Y0dLgtF4IHM) and here: [Google Colab](https://colab.research.google.com/drive/19ulJqMQqk4PfcTx1v3C3cxjvzokrKgZS?usp=sharing)

I have modified the notebook he provides to run locally using Juypter. To run the notebook, you'll need to follow the below instructions on your system. The images I use are From cast photos of Firefly, one of the top 5 sci-fi shows of all time. 
The images are included in the repo, but feel free to choose your own.

## System Setup / Requirements

Note: My system is running Ubuntu 22.0.4, with Python 3.10 installed. I used VS Code to edit/run the files on my system. The extensions I used to assist are PostgreSQL v7.0.4 and Jupyter v2024.1.1

Create the working directory and copy the notebook to it. 

Download the face detection XML file: [https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml](https://github.com/kipr/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
Copy/Move this file to the directory where the notebook will run from. Next, create a subdirectory called 'stored-faces' 

Run: pip install opencv-python, imgbeddings, psycopg2-binary, dotenv 

Install Postgresql (I am using version 14)

Install postgres dev file (postgres.h is required): sudo apt install postgresql-server-dev-14

Install the vector extension found [HERE](https://github.com/pgvector/pgvector): (Instructions from pgvector)
  ```
  cd /tmp
  git clone --branch v0.6.0 https://github.com/pgvector/pgvector.git
  cd pgvector
  make
  make install # required sudo on my system
  ```

Connect to your Postgresql installation using the command: psql

At the default prompt type:
```
defaultdb=>CREATE EXTENSION vector;
defaultdb=>CREATE TABLE pictures (filename TEXT PRIMARY KEY, embedding vector(768));
```

Next, create a .env file for your connection parameters with the following:
```
DB_HOST=127.0.0.1
DB_NAME=pictures
DB_USER=YourUserName
DB_PASSWORD=YourDBPassword
```

I believe that's everything. If you're using VS Code, you should be able to run each cell and have it output both the target image and the found image from the group photo.

I am also working on making this run as a standalone python file (app.py in the repo) using OpenCV to display the target and found images. 
