# Import the required libraries
import cv2
from dotenv import load_dotenv
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load/Store photos -- filenames go here
group_photo = os.path.join(script_dir, "Firefly_Cast_Promo_Fox.jpg")
face_file = os.path.join(script_dir, "cast_one.jpg") 
stored_faces_dir = os.path.join(script_dir, "stored-faces")

# Load classifier
alg = "haarcascade_frontalface_default.xml"
haar_cascade_path = os.path.join(script_dir, alg)
haar_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Read group image
img = cv2.imread(group_photo, 0)
# Create a black and white version of the image
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# Detect the faces
faces = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100)
)

# Loop through the detected faces
i = 0
for x, y, w, h in faces:
    # crop the image to select only the face
    cropped_image = img[y : y + h, x : x + w]
    # loading the target image path into target_file_name variable 
    target_file_name = 'stored-faces/' + str(i) + '.jpg'
    cv2.imwrite(
        target_file_name,
        cropped_image,
    )
    i = i + 1;

# Load environment variables
load_dotenv()

# Retrieve database connection parameters
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

# Create the connection string
conn_str = f"host={db_host} dbname={db_name} user={db_user} password={db_password}"

# Connect to the database
conn = psycopg2.connect(conn_str)

for filename in os.listdir(stored_faces_dir):
    file_path = os.path.join(stored_faces_dir, filename)
    # opening the image
    img = Image.open(file_path)
    # loading the `imgbeddings`
    ibed = imgbeddings()
    # calculating the embeddings
    embedding = ibed.to_embeddings(img)
    cur = conn.cursor()
    cur.execute("INSERT INTO pictures values (%s,%s)", (filename, embedding[0].tolist()))
    print(filename)
conn.commit()


# Open face image
img = Image.open(face_file)
# Load the `imgbeddings`
ibed = imgbeddings()
# Calculate the embeddings
embedding = ibed.to_embeddings(img)

# Find and display match
face_image = cv2.imread(face_file)
cur = conn.cursor()
string_representation = "["+ ",".join(str(x) for x in embedding[0].tolist()) +"]"
cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
rows = cur.fetchall()
if rows:
    # Assuming the first row contains the desired match
    matched_face_filename = rows[0][0]
    # Construct the full path to the image file
    matched_face_image_path = "stored-faces/" + matched_face_filename
    # Load the image
    matched_face_image = cv2.imread(matched_face_image_path)
    # Make sure both images are the same height before combining
    combined_image = cv2.hconcat([face_image, matched_face_image])
    # Add identifying text
    cv2.putText(combined_image, "Face to Find", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined_image, "Matched Face", (face_image.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the combined image
    cv2.imshow("Faces Comparison", combined_image)
    cv2.waitKey(0)  # Wait for a key press to close
    cv2.destroyAllWindows()
else:
    print("No matching images found.")

cur.close()
