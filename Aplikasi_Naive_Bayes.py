import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Title
st.title(
    "Klasifikasi/Prediksi Penyakit Jantung menggunakan Algoritma Klasifikasi Naive Bayes"
)

# Load the heart dataset
heart = pd.read_csv("heart.csv")

# Dataset display
st.subheader("Dataset:")
st.write(heart)

# Description of the dataset
with st.expander("Deskripsi dataset"):
    st.write("1. Age: Usia")
    st.write("2. Sex: 0 = Perempuan, 1 = Laki-laki.")
    st.write(
        "3. Chest Pain (cp) (0-3): Tingkat nyeri dada. 0 = tidak ada nyeri dada, dan 3 = menunjukkan nyeri dada yang tinggi."
    )
    st.write(
        "4. Resting Blood Pressure (trestbps): Tekanan darah. Nilai di atas 120 dianggap tinggi."
    )
    st.write(
        "5. Serum Cholesterol (chol): Kadar kolesterol serum dalam mg/dl. Nilai di bawah 200 dianggap sehat."
    )
    st.write(
        "6. Fasting Blood Sugar (fbs) (0/1): Gula darah setelah puasa. 0 = gula darah rendah < 120, 1 = gula darah tinggi > 120."
    )
    st.write(
        "7. Resting Electrocardiographic Results (restecg) (0-2): Hasil EKG saat istirahat. 0 = rendah, 1 = normal, 2 = tinggi (ada kelainan pada ventrikel kiri)."
    )
    st.write(
        "8. Maximum Heart Rate Achieved (thalach): Denyut jantung maksimum yang dicapai. Semakin tinggi, semakin beresiko terkena penyakit jantung."
    )
    st.write(
        "9. Exercise Induced Angina (exang) (0/1): Kurangnya aliran darah ke jantung karena olahraga atau angin duduk karena olahraga. 0 = tidak, 1 = ya."
    )
    st.write(
        "10. Oldpeak: Depresi ST yang diinduksi oleh latihan relatif terhadap saat istirahat. Menunjukkan tingkat sakit dada yang disebabkan oleh olahraga. Semakin kecil nilai maka semakin buruk."
    )
    st.write(
        "11. Slope (0-2): Penanda iskemia jantung. Semakin ke bawah semakin mengalami penurunan kemampuan otot jantung utk memompa darah. 0: Condong ke atas, 1: Datar, 2: Condong ke bawah."
    )
    st.write(
        "12. Number of Major Vessels (ca) (0-3): Jumlah pembuluh darah besar yang diwarnai dengan flourosopy."
    )
    st.write(
        "13. Thalassemia (thal) (0-3): Kelainan darah. 0: Normal, 1: Cacat tetap, 2: Cacat yang dapat dibalik."
    )
    st.write(
        "14. Target (0/1): Variabel target yang menunjukkan peluang terkena serangan jantung. 0: Tidak terkena penyakit jantung, 1: Terkena penyakit jantung."
    )

# Sidebar for user input
st.sidebar.header("Input Data Testing")
age = st.sidebar.slider("age", min_value=0, max_value=80, value=40)
sex = st.sidebar.selectbox("sex", [0, 1])
cp = st.sidebar.slider("cp", min_value=0, max_value=3, value=1)
trestbps = st.sidebar.slider(
    "trestbps",
    min_value=0,
    max_value=200,
    value=120,
)
chol = st.sidebar.slider("chol", min_value=0, max_value=600, value=240)
fbs = st.sidebar.selectbox(
    "fbs",
    [0, 1],
)
restecg = st.sidebar.selectbox(
    "restecg",
    [0, 1, 2],
)
thalach = st.sidebar.slider("thalach", min_value=0, max_value=220, value=150)
exang = st.sidebar.selectbox(
    "exang",
    [0, 1],
)
oldpeak = st.sidebar.slider(
    "oldpeak",
    min_value=0.0,
    max_value=7.0,
    value=1.0,
)
slope = st.sidebar.slider(
    "slope",
    min_value=0,
    max_value=2,
    value=1,
)
ca = st.sidebar.slider(
    "ca",
    min_value=0,
    max_value=4,
    value=1,
)
thal = st.sidebar.slider("thal", min_value=0, max_value=3, value=2)

# Create a DataFrame for the user input
user_input = pd.DataFrame(
    {
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal],
    }
)

# Gaussian Naive Bayes model training
x = heart.drop(["target"], axis=1)
y = heart["target"]
nbc = GaussianNB()
data_training = nbc.fit(x, y)

# Display actual and predicted values
st.write("Actual vs Predicted:")
st.write(pd.DataFrame({"Actual": y, "Predicted": data_training.predict(x)}))

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train the model
nbc.fit(x_train, y_train)

# Make predictions on the test set
y_pred = nbc.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Nilai akurasi = {accuracy:.2f}")

# Scatter chart to show the relationship between two features
st.subheader("Scatter Chart:")
scatter_fig = px.scatter(
    heart,
    x="trestbps",
    y="chol",
    color="target",
    title="Hubungan antara Tekanan Darah dan Kolesterol",
)
st.plotly_chart(scatter_fig)

# Confusion matrix
st.subheader("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=["Predicted 0", "Predicted 1"],
    y=["Actual 0", "Actual 1"],
    colorscale="Viridis",
)
st.plotly_chart(fig_cm)

# Display user input
st.subheader("Data Test:")
st.write(user_input)

# Make predictions on user input
y_pred = data_training.predict(user_input)

# Display prediction result
st.subheader("Klasifikasi/Prediksi:")
if y_pred == 0:
    result = "Safe"
    st.success("Hasil klasifikasi/prediksi: **Tidak terkena penyakit jantung**")
elif y_pred == 1:
    result = "Risk"
    st.warning("Hasil klasifikasi/prediksi: **Terkena penyakit jantung**")
else:
    result = "Error"
