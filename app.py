import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

st.title("Application de Prédiction Bancaire (Random Forest)")

# Chemin vers CSV sauvegardé
saved_csv_path = "bank (1).csv"

# Choix du mode
mode = st.radio("Choisissez le mode :", 
                ("Charger CSV manuellement", "Afficher CSV sauvegardé", "Faire une prédiction"))

df = None

# Mode 1 : Upload manuel
if mode == "Charger CSV manuellement":
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Aperçu des données chargées")
        st.dataframe(df.head())

# Mode 2 : Affichage du fichier sauvegardé
elif mode == "Afficher CSV sauvegardé":
    if os.path.exists(saved_csv_path):
        df = pd.read_csv(saved_csv_path)
        st.subheader("Contenu du fichier sauvegardé")
        st.dataframe(df.head())
    else:
        st.error(f"Fichier non trouvé : {saved_csv_path}")

# Mode 3 : Prédiction automatique
elif mode == "Faire une prédiction":
    if os.path.exists(saved_csv_path):
        df = pd.read_csv(saved_csv_path)
        st.subheader("Données utilisées pour l'entraînement")
        st.dataframe(df.head())

        # Encodage
        df_encoded = df.copy()
        label_encoders = {}
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le

        target_col = "deposit" if "deposit" in df.columns else "y"
        X = df_encoded.drop(target_col, axis=1)
        y = df_encoded[target_col]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Utilisation de RandomForestClassifier uniquement
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Rapport de Classification")
        st.text(classification_report(y_test, y_pred))

        # Matrice de confusion
        st.subheader("Matrice de Confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        st.pyplot(fig)

        # Courbe ROC (si binaire)
        st.subheader("Courbe ROC (classification binaire uniquement)")
        if len(np.unique(y)) == 2:
            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % roc_auc)
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_xlabel('False Positive Rate')
            ax2.set_ylabel('True Positive Rate')
            ax2.set_title('Courbe ROC')
            ax2.legend(loc="lower right")
            st.pyplot(fig2)

        # Formulaire pour prédiction personnalisée
        st.subheader("Prédiction Personnalisée")

        sample = []
        for col in X.columns:
            if col in categorical_cols:
                options = label_encoders[col].classes_
                choice = st.selectbox(f"{col}", options)
                val = label_encoders[col].transform([choice])[0]
                sample.append(val)
            else:
                val = st.number_input(f"{col}", value=float(df[col].mean()))
                sample.append(val)

        if st.button("Prédire"):
            sample_np = np.array(sample).reshape(1, -1)
            sample_np = scaler.transform(sample_np)
            prediction = model.predict(sample_np)
            if target_col in label_encoders:
                label = label_encoders[target_col].inverse_transform(prediction)[0]
            else:
                label = prediction[0]
            st.success(f"Résultat de la prédiction : {label}")
    else:
        st.error(f"Fichier {saved_csv_path} introuvable. Chargez un fichier CSV d'abord.")
