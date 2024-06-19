# Projet Entrainement d'une IA

Ce projet utilise FastAPI pour créer une API de classification d'images et Streamlit pour créer une interface utilisateur. Le modèle de classification utilise MobileNetV2 pré-entraîné pour obtenir des performances optimales.

## Prérequis

- Python 3.7 ou plus récent
- pip

## Installation

1. Clonez ce dépôt :

    ```bash
    git clone https://github.com/votre-utilisateur/votre-repo.git
    cd votre-repo
    ```

2. Créez un environnement virtuel et activez-le :

    ```bash
    python -m venv env
    # Sur Windows
    .\env\Scripts\activate
    # Sur macOS/Linux
    source env/bin/activate
    ```

3. Installez les dépendances :

    ```bash
    pip install -r requirements.txt
    ```

## Démarrage de l'API FastAPI

1. Assurez-vous que l'environnement virtuel est activé.
2. Lancez le serveur FastAPI :

    ```bash
    uvicorn api:app --reload
    ```

L'API sera disponible à l'adresse `http://127.0.0.1:8000`.

## Démarrage de l'Interface Streamlit

1. Dans un nouveau terminal (assurez-vous que l'environnement virtuel est activé), lancez Streamlit :

    ```bash
    streamlit run app.py
    ```

L'interface utilisateur sera disponible à l'adresse `http://localhost:8501`.

## Utilisation

### Entraînement du Modèle

1. Accédez à l'interface Streamlit.
2. Entrez les noms des classes (par exemple, "chat" et "chien").
3. Téléchargez les images pour chaque classe.
4. Cliquez sur "Entraîner le Modèle".
5. L'API entraînera le modèle et affichera la précision obtenue.

### Classification d'Images

1. Accédez à l'interface Streamlit.
2. Téléchargez exactement deux images à classer.
3. L'API retournera les classifications des images.

## Structure du Projet

```plaintext
.
├── api.py                   # Code de l'API FastAPI
├── app.py                   # Interface utilisateur Streamlit
├── requirements.txt         # Dépendances du projet
├── models/                  # Dossier pour les modèles sauvegardés
│   ├── model.h5             # Modèle entraîné
│   └── label_encoder.joblib # Encodeur d'étiquettes
├── .gitignore               # Fichiers et dossiers à ignorer par Git
└── README.md                # Instructions du projet
