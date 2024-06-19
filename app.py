import streamlit as st
import requests

# URL de l'API
API_URL_TRAIN = "http://127.0.0.1:8000/train"
API_URL_CLASSIFY = "http://127.0.0.1:8000/classify_images"

st.title("Application de Classification d'Images")

st.header("Entraîner le Modèle")
with st.form(key='train_form'):
    class1_name = st.text_input("Entrez le nom de la première classe")
    class2_name = st.text_input("Entrez le nom de la deuxième classe")
    
    class1_files = st.file_uploader(f"Choisissez des images pour la classe: {class1_name}", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="class1")
    class2_files = st.file_uploader(f"Choisissez des images pour la classe: {class2_name}", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="class2")
    
    train_button = st.form_submit_button(label='Entraîner le Modèle')

    if train_button:
        if class1_files and class2_files and class1_name and class2_name:
            files = []
            labels = []
            
            for file in class1_files:
                files.append(("files", (file.name, file, file.type)))
                labels.append(class1_name)
            
            for file in class2_files:
                files.append(("files", (file.name, file, file.type)))
                labels.append(class2_name)
            
            response = requests.post(API_URL_TRAIN, files=files, data={"labels": ",".join(labels)})
            if response.status_code == 200:
                result = response.json()
                st.success(f"Modèle entraîné avec succès avec une précision de: {result['accuracy']:.2f}")
            else:
                st.error(f"Erreur lors de l'entraînement du modèle: {response.json()['message']}")
        else:
            st.error("Veuillez fournir des noms pour les deux classes et télécharger des images pour chaque classe.")

st.header("Classer des Images")
uploaded_files = st.file_uploader("Choisissez deux images à classer", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) == 2:
    files = [("files", (file.name, file, file.type)) for file in uploaded_files]
    response = requests.post(API_URL_CLASSIFY, files=files)
    if response.status_code == 200:
        results = response.json()["results"]
        for result in results:
            st.write(f"Image `{result['filename']}` classée comme `{result['classification']}`")
    else:
        st.error(f"Erreur lors de la classification des images: {response.json()['message']}")
else:
    if uploaded_files:
        st.error("Veuillez télécharger exactement deux images.")
