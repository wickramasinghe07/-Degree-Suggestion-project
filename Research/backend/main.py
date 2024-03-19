from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import joblib
import numpy as np
import cv2
import mediapipe as mp
import time
import pyrebase
from openai import OpenAI
import spacy
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"], 
    allow_headers=["*"], 
    allow_credentials=True,
)


firebaseConfig = {
    "apiKey": "AIzaSyAcJPhbBGhQlSvOLkWC5ccpRIkpitlOEbc",
    "authDomain": "uxdb-b0a22.firebaseapp.com",
    "databaseURL": "https://uxdb-b0a22-default-rtdb.firebaseio.com",
    "projectId": "uxdb-b0a22",
    "storageBucket": "uxdb-b0a22.appspot.com",
    "messagingSenderId": "459246476650",
    "appId": "1:459246476650:web:1d355243c53a6f0af19913",
    "measurementId": "G-H34SCZZRMC"
}

client = OpenAI(api_key="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")

firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=1)


@app.get("/show_cam")
async def show_cam():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    state = "Forward1"  # Initial state
    state_start_time = time.time()
    state_durations = {"Looking Left": 0, "Looking Right": 0, "Forward": 0}

    while cap.isOpened():
        success, image = cap.read()

        start = time.time()

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # flipped for selfie view

        image.flags.writeable = False

        results = face_mesh.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_2d = []
        face_3d = []

        # closeOrOpen = db.child("handelCam").get().val()
        # if closeOrOpen == 0:
        #     break
        
        if results.multi_face_landmarks:
            closeOrOpen = db.child("handelCam").get().val()
            if closeOrOpen == 0:
                break
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])
                        face_3d.append(([x, y, lm.z]))

                # Get 2d Coord
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

                # getting rotational of face
                rmat, jac = cv2.Rodrigues(rotation_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # add this dgree programs ,these are changing 
                # degreePrograms = ['sliit_DS', 'sliit_CS', 'sliit_SE'] 
                degreePrograms = db.child("currentDegreeData").get().val()
                
                # here based on axis rot angle is calculated
                if y < -5:
                    new_state = "Looking Left"
                elif y > 5:
                    new_state = "Looking Right"
                else:
                    new_state = "Forward"

                if new_state != state:
                    state = new_state
                    # state_durations[state] += time.time() - state_start_time
                    state_durations[state] += round(time.time() - state_start_time, 2)
                    if new_state == "Looking Left" :
                        print(state_durations[state])
                        db.child("testSet").update(data={degreePrograms[0]:state_durations[state]})
                    if new_state == "Looking Right" :
                        print(state_durations[state])
                        db.child("testSet").update(data={degreePrograms[2]:state_durations[state]})
                    if new_state == "Forward" :
                        print(state_durations[state])
                        db.child("testSet").update(data={degreePrograms[1]:state_durations[state]})
                    state_start_time = time.time()
                    state = new_state

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix,
                                                                distortion_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(image, p1, p2, (255, 0, 0), 3)

                cv2.putText(image, state, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        # cv2.imshow('Head Pose Detection', image)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     break

    # Update duration for the last state
    state_durations[state] += time.time() - state_start_time

    print("Duration of each state:")
    for state, duration in state_durations.items():
        print(f"{state}: {duration} seconds")

    cap.release()
    cv2.destroyAllWindows()

    return {"message": "Webcam feed closed"}


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

# my particulars are as follows:
# Name: Paranga

class InputCourseName(BaseModel):
    courseName: str

nlp = spacy.load("en_core_web_sm")

@app.post("/course")
async def getcourse(cn:InputCourseName):
    doc = nlp(cn.courseName)

    keywords = [
    "computer", "information", "IT", "course", "computer science", "information technology", 
    "programming", "software", "development", "coding", "computer programming", "web development",
    "networking", "cybersecurity", "artificial intelligence", "machine learning",
    "data science", "software engineering", "database", "algorithm", "computer engineering",
    "informatics", "computer architecture", "systems programming", "cloud computing",
    "mobile development", "operating systems", "object-oriented programming",
    "functional programming", "computer graphics", "internet of things", "big data",
    "computer vision", "natural language processing", "blockchain", "digital forensics",
    "game development", "human-computer interaction", "computer ethics", "parallel computing",
    "distributed systems", "information retrieval", "computational biology", "bio informatics",
    "network security", "computer networks", "embedded systems", "software testing",
    "computer hardware", "system administration", "computer vision", "information security",
    "programming languages", "data mining", "computer algebra", "quantum computing",
    "computer-based education", "computational linguistics", "e-commerce",
    "computational intelligence", "computer-assisted design", "scientific computing",
    "high-performance computing", "computational physics", "computational chemistry",
    "computational neuroscience", "computational economics", "computational sociology",
    "computational music", "computational art"
    ]

    for token in doc:
        if token.text.lower() in keywords:
            return {"courseRes": "Yes, I have taken several IT-related courses and find them beneficial."}
    return {"courseRes": "No, I haven't taken any IT-related courses yet, but I'm considering it."}

# @app.post("/course")
# async def getcourse(cn:InputCourseName):
#     promt = f"i have question : Do you have any experience with IT-related courses? i did course {cn.courseName}.  answers to select : Yes, I have taken several IT-related courses and find them beneficial No, I haven't taken any IT-related courses yet, but I'm considering it.Yes, but I didn't find them very useful for my career goals.No, I don't believe IT-related courses are necessary for my field of interest what shoude i select "

#     completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "user", "content": promt}
#     ]
#     )

#     return {"courseRes":completion.choices[0].message.content}

class InputInterestIn(BaseModel):
    InterestIn: str


@app.post("/course_cost")
async def getcourseValue(ii:InputInterestIn):
    promt = f"I have Interest In doing {ii.InterestIn} degree , I have question : Do you consider the cost and potential value when deciding on learning opportunities?  . answers to select :Yes, I carefully weigh the cost against the potential value and benefits. , No, I prioritize learning opportunities solely based on their relevance and content,Yes, but I tend to prioritize lower-cost options even if they may have less value. , No, I believe that investing in learning is essential regardless of the cost."
    
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": promt}
    ]
    )
    out = "I suggest to select : " + completion.choices[0].message.content

    return {"valueRes":out}

class InputUserDetails(BaseModel):
    InterestIn: str
    AlStream: str

skills_and_languages_to_check = {
    "analytical thinking", "problem-solving", "mathematical reasoning",
    "Python", "R", "Java", "C/C++", "JavaScript", "Swift", "Kotlin", "HTML/CSS",
    "SQL", "NoSQL", "Frontend Development", "Backend Development", "Full-stack Development",
    "React", "Angular", "Vue.js", "Django", "Flask", "iOS Development", "Android Development",
    "Cross-platform Development", "Flutter", "React Native", "Data Analysis",
    "Data Visualization Tools", "Statistical Analysis", "Machine Learning Algorithms",
    "Deep Learning", "Natural Language Processing", "Computer Vision",
    "Object-Oriented Programming", "Version Control", "Agile Methodologies",
    "Test-Driven Development", "Continuous Integration/Continuous Deployment",
    "Network Security", "Cryptography", "Ethical Hacking", "Secure Coding Practices",
    "Problem-solving Skills", "Algorithm Design and Analysis", "Data Structures",
    "Verbal and Written Communication Skills", "Teamwork and Collaboration", "Presentation Skills"
}



@app.post("/skills") 
async def getskillValue(ud:InputUserDetails):
    promt = f"my higher education stream is {ud.AlStream} ,I'm interest in {ud.InterestIn} ,based on my higher education Stream and interest  what I might have skills ,give only skills name  'skill,skill,skill' this format  ,nothing else "
    # promt = f"my higher education stream is {ud.AlStream} ,I'm interest in {ud.InterestIn} ,based on my higher education Stream and interest  what I might have skills ,need one string output"
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": promt}
    ]
    )
    matching_skills = set()

    for skill in skills_and_languages_to_check:
        skill_tokens = word_tokenize(skill.lower())
        if any(token in completion.choices[0].message.content for token in skill_tokens):
            matching_skills.add(skill)

    skills_and_languages_string = ', '.join(matching_skills)
    out = "You might have  : " + skills_and_languages_string
    return {"valueRes":out}

# @app.post("/skills")
# async def getskillValue(ud:InputUserDetails):
#     # promt = f"my higher education stream is {ud.AlStream} ,I'm interest in {ud.InterestIn} ,based on my higher education Stream and interest  what I might have skills ,give only skills name  'skill,skill,skill' this format  ,nothing else "
#     promt = f"my higher education stream is {ud.AlStream} ,I'm interest in {ud.InterestIn} ,based on my higher education Stream and interest  what I might have skills "
#     completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "user", "content": promt}
#     ]
#     )
    
#     out = "You might have  : " + completion.choices[0].message.content 

#     return {"valueRes":out}

class InterestSkillMapper:
    def __init__(self, interest_embeddings, skill_embeddings, interests, skills):
        self.interest_embeddings = interest_embeddings
        self.skill_embeddings = skill_embeddings
        self.interests = interests
        self.skills = skills

    def get_skills_for_interest(self, interest):
        if interest not in self.interests:
            return []

        interest_index = self.interests.index(interest)
        interest_embedding = self.interest_embeddings[interest_index].reshape(1, -1)
        similarities = cosine_similarity(interest_embedding, self.skill_embeddings)
        most_similar_skill_index = np.argmax(similarities)
        return self.skills[most_similar_skill_index]

    def check_interest_exists(self, interest):
        return interest in self.interests


interest_embeddings = np.random.rand(5, 100)
skill_embeddings = np.random.rand(5, 100)

interests = ["AI", "Data Science", "Software Engineering", "Machine Learning", "Network Security", "Robotics", "Natural Language Processing", "Cybersecurity", "Computer Vision"]

skills = [
    ["Machine Learning Algorithms", "Deep Learning", "Natural Language Processing", "Data Analysis", "TensorFlow", "PyTorch", "Scikit-learn"],
    ["Data Analysis", "Data Visualization Tools", "Statistical Analysis", "Python (pandas, numpy, matplotlib)", "R", "Tableau"],
    ["Python", "Java", "C/C++", "JavaScript", "Version Control", "Agile Methodologies", "Test-Driven Development"],
    ["Python", "R", "Machine Learning Algorithms", "Deep Learning", "Statistical Analysis"],
    ["Network Security", "Cryptography", "Ethical Hacking", "Secure Coding Practices", "Wireshark", "Nmap", "Metasploit"],
    ["Robotics Programming", "Control Systems", "ROS (Robot Operating System)", "Computer Vision", "Embedded Systems"],
    ["Natural Language Processing", "Text Mining", "Chatbot Development", "Word Embeddings", "NLTK", "SpaCy"],
    ["Cybersecurity Fundamentals", "Penetration Testing", "Digital Forensics", "Security Protocols", "Firewalls"],
    ["Computer Vision", "Image Processing", "Object Detection", "OpenCV", "Deep Learning for Computer Vision"]
]


class InputInterestInDegree(BaseModel):
    InterestInDegree: str

@app.post("/InterestDegree")
async def getInterestDegree(id:InputInterestInDegree):
    interest_skill_mapper = InterestSkillMapper(interest_embeddings, skill_embeddings, interests, skills)
    if interest_skill_mapper.check_interest_exists(id.InterestInDegree):
        relevant_skills = interest_skill_mapper.get_skills_for_interest(id.InterestInDegree)
        out = f"To select {id.InterestInDegree} you might need : " + relevant_skills
        return {"valueRes":out}
    else:
        promt = f"I am interested in {id.InterestInDegree} degree field , I want to know what are the skills I should have to start this degree . give only 5 skills in this format 'skill, ,skill'  "
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": promt}
        ]
        )
        out = f"To select {id.InterestInDegree} you might need : " + completion.choices[0].message.content

        return {"valueRes":out}

    

    