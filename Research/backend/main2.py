from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
from pydantic import BaseModel
import os
import joblib
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"], 
    allow_headers=["*"], 
    allow_credentials=True,
)


def generate_frames():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        # Display the frame using OpenCV
        cv2.imshow('Webcam Feed', frame)
        cv2.waitKey(1)  # Wait for 1 millisecond

        # Convert frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break
        # Yield the frame in bytes
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

@app.get("/show_cam")
async def show_cam():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        # Display the frame using OpenCV
        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to quit
            break

    # Release the webcam and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

    return {"message": "Webcam feed closed"}

@app.get("/video_feed")
async def video_feed():
    # Return a StreamingResponse object with content_type "multipart/x-mixed-replace"
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


class InputData(BaseModel):
    data: list

class OutputData(BaseModel):
    prediction: str

# Load the trained models
save_dir = "saved_models_d"
classifiers = []

for model_file in os.listdir(save_dir):
    if model_file.endswith("_model.joblib"):
        model_path = os.path.join(save_dir, model_file)
        classifier = joblib.load(model_path)
        classifiers.append(classifier)

disorder_labels = {
    0: "SE",
    1: "DS",
    2: "IT",
    3: "ICT",
    4: "CN",
    5: "CS"
}

input_mapping = {'Maths': 0, 'Science': 1, 'Art': 2, 'Tech': 3,'IT': 0, 'AI': 1, 'SE': 2, 'CS': 3}


@app.post("/predict")
async def predict(data: InputData):
    print(data)
    # input_array = np.array(data.data).reshape(1, -1)
    numeric_data = [input_mapping[item] if item in input_mapping else item for item in data.data]

    converted_list = [int(x) if isinstance(x, str) else x for x in numeric_data]


    print(numeric_data)
    input_array = np.array(converted_list).reshape(1, -1)

    # Perform predictions using all loaded models
    predictions = [classifier.predict(input_array)[0] for classifier in classifiers]
    print(predictions)
    
    # Perform majority voting for the ensemble
    ensemble_prediction = max(set(predictions), key=predictions.count)

    predicted_label = disorder_labels.get(ensemble_prediction, "Unknown Disorder")

    print(predicted_label)

    return {"predicted_label":predicted_label}


# from fastapi import FastAPI, Response
# from fastapi.responses import StreamingResponse
# from typing import Optional
# import cv2

# app = FastAPI()

# stop_streaming = False

# def generate_frames(stop_streaming):
#     # Open the webcam
#     cap = cv2.VideoCapture(0)
#     while True:
#         if stop_streaming:
#             break
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # Display the frame using OpenCV
#         cv2.imshow('Webcam Feed', frame)
#         cv2.waitKey(1)  # Wait for 1 millisecond

#         # Convert frame to JPEG format
#         ret, buffer = cv2.imencode('.jpg', frame)
#         if not ret:
#             break
#         # Yield the frame in bytes
#         yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

#     # Release the webcam and close the window
#     cap.release()
#     cv2.destroyAllWindows()

# @app.get("/video_feed")
# async def video_feed(stop: Optional[bool] = False):
#     global stop_streaming
#     stop_streaming = stop
#     # Return a StreamingResponse object with content_type "multipart/x-mixed-replace"
#     return StreamingResponse(generate_frames(stop), media_type="multipart/x-mixed-replace; boundary=frame")

