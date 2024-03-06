# Import the required libraries
import cv2
from dotenv import load_dotenv
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os

# Get the directory of app.py
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
# Detect the faces
faces = haar_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))
# Loop through the detected faces
i = 0
for x, y, w, h in faces:
    # crop the image to select only the face and create file in 'stored-faces' directory
    cropped_image = img[y : y + h, x : x + w]
    target_file_name = os.path.join(stored_faces_dir, str(i) + '.jpg')
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
find_img = Image.open(face_file)
# Load the `imgbeddings`
ibed = imgbeddings()
# Calculate the embeddings
find_embedding = ibed.to_embeddings(find_img)

# Find and display match
face_image = cv2.imread(face_file)
cur = conn.cursor()
string_representation = "["+ ",".join(str(x) for x in find_embedding[0].tolist()) +"]"
cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
rows = cur.fetchall()
if rows:
    matched_face_filename = rows[0][0]
    matched_face_image_path = os.path.join(stored_faces_dir, matched_face_filename)
    matched_face_image = cv2.imread(matched_face_image_path)

    # Make sure both images are the same height before combining
    if face_image.shape[0] != matched_face_image.shape[0]:
        new_height = min(face_image.shape[0], matched_face_image.shape[0])
        face_image = cv2.resize(face_image, (int(face_image.shape[1] * (new_height / face_image.shape[0])), new_height))
        matched_face_image = cv2.resize(matched_face_image, (int(matched_face_image.shape[1] * (new_height / matched_face_image.shape[0])), new_height))

    separator_width = 30
    separator = np.zeros((face_image.shape[0], separator_width, 3), dtype=np.uint8)
    combined_image = np.hstack((face_image, separator, matched_face_image))
    # combined_image = cv2.hconcat([face_image, matched_face_image])
    # Add identifying text
    cv2.putText(combined_image, "Face to Find", (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined_image, "Matched Face", (face_image.shape[1] + 20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the combined image
    cv2.imshow("Faces Comparison", combined_image)
    cv2.waitKey(0)  # Wait for a key press to close
    cv2.destroyAllWindows()
else:
    print("No matching images found.")

# Cleanup rows from the pictures table before closing the database connection
cur.execute("DELETE FROM pictures;")
conn.commit()

# Clear the stored-faces directory
for filename in os.listdir(stored_faces_dir):
    file_path = os.path.join(stored_faces_dir, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)

# Close the database connection
cur.close()
conn.close()
