import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2

from PIL import Image
from datetime import datetime

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

import av

# =====================
# CONFIG
# =====================

st.set_page_config(
    page_title="Moodify",
    layout="wide"
)

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* =====================
BACKGROUND
===================== */

.stApp{
    background:#fff3e6;
}

/* =====================
HOME TITLE
===================== */

.big-title{
    text-align:center;
    font-size:120px;
    font-weight:900;
    color:#ff8435;
    margin-bottom:-20px;
    font-family:'Segoe UI',sans-serif;
}

.sub-title{
    text-align:center;
    font-size:42px;
    font-weight:800;
    color:#2d3748;
    font-family:'Segoe UI',sans-serif;
}

.desc{
    text-align:center;
    font-size:24px;
    font-weight:500;
    color:#666666;
    max-width:900px;
    margin:auto;
    line-height:1.8;
    font-family:'Segoe UI',sans-serif;
}

.line{
    width:80px;
    height:6px;
    background:#ff8435;
    margin:25px auto;
    border-radius:20px;
}

.feature-card{

    background:white;

    padding:25px;

    border-radius:24px;

    text-align:center;

    box-shadow:0 8px 20px rgba(0,0,0,0.06);

    min-height:140px;

    transition:0.3s;
}

.feature-card:hover{

    transform:translateY(-5px);

    box-shadow:0 12px 30px rgba(255,132,53,0.15);
}

.feature-card h3{

    color:#2d3748;

    font-size:24px;

    font-weight:800;

    margin-bottom:10px;
}

.feature-card p{

    color:#666;

    font-size:16px;
}

.stApp{
    background:linear-gradient(
        180deg,
        #fffaf5 0%,
        #fff3e6 100%
    );
}

/* =====================
SECTION TITLE
===================== */

.section{
    font-size:42px;
    font-weight:900;
    color:#2d3748;
    font-family:'Segoe UI',sans-serif;
}

/* =====================
NAVBAR STYLE
===================== */

.nav-wrapper{

    background:white;

    padding:12px;

    border-radius:24px;

    box-shadow:0 8px 20px rgba(0,0,0,0.08);

    margin-bottom:40px;
}

div.stButton > button{

    width:100%;

    background:linear-gradient(
        135deg,
        #ff7a1a,
        #ff9b55
    );

    color:white;

    border:none;

    border-radius:18px;

    height:65px;

    font-size:24px;

    font-weight:800;

    font-family:'Segoe UI',sans-serif;

    transition:0.3s;

    box-shadow:0 4px 12px rgba(
        255,
        132,
        53,
        0.25
    );
}

div.stButton > button:hover{

    transform:translateY(-2px);

    background:linear-gradient(
        135deg,
        #ff8a35,
        #ffab70
    );

    box-shadow:0 8px 20px rgba(
        255,
        132,
        53,
        0.35
    );
}

# =====================
# NAVIGATION
# =====================

if "page" not in st.session_state:
    st.session_state.page = "Home"

col1,col2,col3 = st.columns(3)

with col1:
    if st.button(
        "🏠 Home",
        use_container_width=True
    ):
        st.session_state.page = "Home"

with col2:
    if st.button(
        "📸 Emotion Scan",
        use_container_width=True
    ):
        st.session_state.page = "Scan"

with col3:
    if st.button(
        "📅 Mood Calendar",
        use_container_width=True
    ):
        st.session_state.page = "Riwayat"
        
/* =====================
FILE UPLOADER
===================== */

[data-testid="stFileUploader"]{

    background:white;

    border-radius:20px;

    padding:15px;

    border:2px dashed #ff8435;
}

/* =====================
RESULT BOX
===================== */

.result-box{

    background:white;

    padding:35px;

    border-radius:25px;

    box-shadow:0px 8px 20px rgba(0,0,0,0.08);

    text-align:center;
}

.result-box h1{

    color:#ff8435;

    font-size:52px;

    font-weight:900;

    margin-bottom:10px;
}

.result-box h3{

    color:#2d3748;

    font-size:24px;

    font-weight:700;
}

/* =====================
CALENDAR BOX
===================== */

.calendar-box{

    background:white;

    border-radius:20px;

    box-shadow:0px 4px 10px rgba(0,0,0,0.06);
}

/* =====================
IMAGE CENTER
===================== */

[data-testid="stImage"]{
    text-align:center;
}

/* =====================
SELECTBOX
===================== */

.stSelectbox label{

    font-size:18px;

    font-weight:700;

    color:#2d3748;
}

</style>
""", unsafe_allow_html=True)

# =====================
# MODEL
# =====================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "emotion_model.h5"
    )

model = load_model()
IMG_SIZE = model.input_shape[1]
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

score_map = {
    "Happy":100,
    "Surprise":90,
    "Neutral":70,
    "Sad":40,
    "Fear":30,
    "Angry":20,
    "Disgust":10
}

class EmotionProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
        )

        for (x,y,w,h) in faces:

            face = img[y:y+h, x:x+w]

            try:

                face_input = cv2.resize(
                    face,
                    (IMG_SIZE, IMG_SIZE)
                )

                face_input = (
                    face_input.astype(np.float32)
                    / 255.0
                )

                face_input = np.expand_dims(
                    face_input,
                    axis=0
                )

                prediction = model.predict(
                    face_input,
                    verbose=0
                )

                emotion = emotion_labels[
                    np.argmax(prediction)
                ]

                confidence = np.max(
                    prediction
                )

                cv2.rectangle(
                    img,
                    (x,y),
                    (x+w,y+h),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    img,
                    f"{emotion} {confidence:.0%}",
                    (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2
                )

            except:
                pass

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )
        
# =====================
# NAVIGATION
# =====================

if "page" not in st.session_state:
    st.session_state.page = "Home"

st.markdown(
'<div class="nav-container">',
unsafe_allow_html=True
)

col1,col2,col3 = st.columns(3)

with col1:
    if st.button("Home", use_container_width=True):
        st.session_state.page = "Home"

with col2:
    if st.button("Emotion Scan", use_container_width=True):
        st.session_state.page = "Scan"

with col3:
    if st.button("Mood Calendar", use_container_width=True):
        st.session_state.page = "Riwayat"

st.markdown(
'</div>',
unsafe_allow_html=True
)

st.write("")

# =====================
# HOME
# =====================

if st.session_state.page == "Home":

    st.markdown("<br>", unsafe_allow_html=True)

    # Logo
    c1,c2,c3 = st.columns([4.2,1,4])

    with c2:
        st.image(
            "Untitled design (11).png",
            width=150
        )

    st.markdown(
        """
        <div class="big-title">
        Moodify
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sub-title">
        AI-Based Learning Engagement Analytics
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="line"></div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="desc">
        Understand your emotions, improve your learning experience,
        and track your progress every day.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    c1,c2,c3 = st.columns([3,2,3])

    with c2:
        if st.button(
            "Start Scanning →",
            use_container_width=True
        ):
            st.session_state.page = "Scan"
            st.rerun()

# =====================
# SCAN
# =====================

elif st.session_state.page == "Scan":

    st.markdown(
        '<div class="section">Upload File</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "",
        type=["jpg","jpeg","png"]
    )

    st.write("")

    st.markdown(
        '<div class="section">Kamera</div>',
        unsafe_allow_html=True
    )

    image = None

    # =====================
    # KAMERA FOTO
    # =====================

    use_photo_camera = st.toggle(
        "📸 Aktifkan Kamera Foto"
    )

    if use_photo_camera:

        camera = st.camera_input(
            "Ambil Foto"
        )

        if camera:
            image = Image.open(camera)

    # =====================
    # KAMERA REALTIME
    # =====================

    use_realtime = st.toggle(
        "🎥 Aktifkan Kamera Real-Time"
    )

    if use_realtime:

        webrtc_streamer(
            key="emotion-realtime",
            video_processor_factory=
            EmotionProcessor,
            media_stream_constraints={
                "video": True,
                "audio": False
            }
        )

    # =====================
    # UPLOAD FOTO
    # =====================

    if uploaded_file:
        image = Image.open(uploaded_file)

    st.write("")

    st.markdown(
        '<div class="section">Hasil Deteksi</div>',
        unsafe_allow_html=True
    )

    if image:

        img_np = np.array(image)

        gray = cv2.cvtColor(
            img_np,
            cv2.COLOR_RGB2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5
        )

        if len(faces) > 0:

            x,y,w,h = faces[0]
            
            preview = img_np.copy()

            cv2.rectangle(
                preview,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                3
            )

            st.image(
                preview,
                caption="Face Detected",
                width=350
            )

            face = img_np[
                y:y+h,
                x:x+w
            ]

            face = cv2.resize(
            face,
            (IMG_SIZE, IMG_SIZE)
            )

            face = face / 255.0

            face = np.expand_dims(
                face,
                axis=0
            )

            prediction = model.predict(face)

        else:

            st.error(
                "Wajah tidak terdeteksi"
            )

            st.stop()

        predicted_class = np.argmax(
            prediction
        )

        confidence = np.max(
            prediction
        )

        emotion = emotion_labels[
            predicted_class
        ]

        score = score_map[
            emotion
        ]

        st.session_state.result = {
            "emotion":emotion,
            "confidence":confidence,
            "score":score
        }

        history_file = "history.csv"

        new_data = pd.DataFrame([{
            "datetime":datetime.now(),
            "emotion":emotion,
            "confidence":float(confidence),
            "score":score
        }])

        if os.path.exists(history_file):

            old = pd.read_csv(
                history_file
            )

            history = pd.concat(
                [old,new_data],
                ignore_index=True
            )

        else:

            history = new_data

        history.to_csv(
            history_file,
            index=False
        )

        st.success(
            "Deteksi berhasil"
        )

        st.write("")
        st.write("")

        st.markdown(
            '<div class="section">Detection Result</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result-box">

            <h1>{emotion}</h1>

            <h3>Confidence :
            {confidence:.2%}</h3>

            <h3>Moodify Score :
            {score}</h3>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        st.write("")

        st.markdown(
            '<div class="section">Activity Recommendation</div>',
            unsafe_allow_html=True
        )

        recommendations = {

            "Happy":[
                "Menyusun daftar tugas harian", 
                "Mempelajari hal baru", 
                "Olahraga ringan"
            ],

            "Neutral":[
                "Membaca buku",
                "Journaling refleksi harian"
            ],

            "Sad":[
                "Meditasi singkat",
                "Menghubungi teman dekat",
                "Berjalan santai"
            ],

            "Fear":[
                "Teknik grounding 5-4-3-2-1",
                "Pernapasan kotak (box breathing)"
            ],

            "Angry":[
                "Latihan pernapasan",
                "Mendengarkan musik instrumental",
                "Journaling"
            ],

            "Disgust":[
                "Mengalihkan perhatian sejenak",
                "Peregangan ringan"
            ],

            "Surprise":[
                "Mencatat momen",
                "Refleksi singkat"
            ]
        }

        for item in recommendations[emotion]:

            st.write(
                f"✅ {item}"
            )

# =====================
# RIWAYAT
# =====================

elif st.session_state.page == "Riwayat":

    import calendar

    st.markdown(
        '<div class="section">Mood Calendar</div>',
        unsafe_allow_html=True
    )

    bulan_list = [
        "Januari","Februari","Maret","April",
        "Mei","Juni","Juli","Agustus",
        "September","Oktober","November","Desember"
    ]

    col1,col2 = st.columns(2)

    with col1:
        bulan = st.selectbox(
            "Bulan",
            bulan_list
        )

    with col2:
        tahun = st.selectbox(
            "Tahun",
            [2025,2026,2027]
        )

    mood_data = {}

    if os.path.exists("history.csv"):

        history = pd.read_csv("history.csv")

        history["datetime"] = pd.to_datetime(
            history["datetime"]
        )

        for _, row in history.iterrows():

            tanggal = row["datetime"].date()

            mood_data[str(tanggal)] = row["emotion"]

    emoji_map = {

        "Happy":"😊",
        "Neutral":"😐",
        "Sad":"😢",
        "Angry":"😠",
        "Fear":"😨",
        "Disgust":"🤢",
        "Surprise":"😲"
    }

    month_num = bulan_list.index(
        bulan
    ) + 1

    cal = calendar.monthcalendar(
        tahun,
        month_num
    )

    hari = [
        "Sen","Sel","Rab",
        "Kam","Jum","Sab","Min"
    ]

    cols = st.columns(7)

    for i,h in enumerate(hari):

        cols[i].markdown(
            f"### {h}"
        )

    for week in cal:

        cols = st.columns(7)

        for i,day in enumerate(week):

            if day == 0:

                cols[i].write("")

            else:

                tanggal_str = (
                    f"{tahun}-{month_num:02d}-{day:02d}"
                )

                emoji = ""

                if tanggal_str in mood_data:

                    emotion = mood_data[
                        tanggal_str
                    ]

                    emoji = emoji_map.get(
                        emotion,
                        ""
                    )

                cols[i].markdown(
                    f"""
                    <div style="
                    background:#ffe0c7;
                    border-radius:20px;
                    height:110px;
                    margin:8px;
                    display:flex;
                    flex-direction:column;
                    justify-content:center;
                    align-items:center;
                    ">
                    <div style="
                    font-size:24px;
                    font-weight:bold;
                    ">
                    {day}
                    </div>

                    <div style="
                    font-size:28px;
                    ">
                    {emoji}
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )