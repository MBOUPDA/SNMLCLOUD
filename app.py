# =============================================================================
# GROUPE 3 - SN_ML_CLOUD_MBOUPDA_JOYCE
# Streamlit Multi-Feature ML Application
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
import io
import tempfile
import base64

warnings.filterwarnings("ignore")

# ======================== PAGE CONFIG ========================
st.set_page_config(
    page_title="GROUPE 3 — ML Cloud App",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    text-align: center;
    padding: 1.2rem 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.5);
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header">🤖 GROUPE 3 — ML Cloud Application</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ======================== TABS ========================
tabs = st.tabs([
    "🏠 House Price Regression",
    "🩺 Diabetes Classification",
    "📄 RAG on PDF",
    "🎵 Audio Processing",
    "📊 LangSmith Monitoring",
    "📈 LSTM Stock Prediction",
    "🔮 HF Time Forecasting",
])

# ============================================================
#  TAB 1 — HOUSE PRICE REGRESSION (ANN)
# ============================================================
with tabs[0]:
    st.header("🏠 House Price Prediction — Artificial Neural Network")
    st.markdown(
        "**Dataset :** [House Data — Kaggle](https://www.kaggle.com/datasets/shree1992/housedata)  \n"
        "Prédire le prix des maisons avec un ANN multi-couche TensorFlow/Keras."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Configuration")
        house_file = st.file_uploader("Upload house data CSV", type=["csv"], key="house_data")
        if house_file is None:
            st.info("Données synthétiques utilisées si aucun fichier n'est chargé.")

        h_epochs   = st.slider("Epochs", 10, 200, 50, key="h_epochs")
        h_batch    = st.selectbox("Batch Size", [16, 32, 64, 128], index=1, key="h_batch")
        h_layers   = st.multiselect("Hidden Layers (unités)", [32, 64, 128, 256, 512],
                                     default=[128, 64], key="h_layers")
        h_lr       = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01, 0.1],
                                       value=0.001, key="h_lr")
        h_dropout  = st.slider("Dropout Rate", 0.0, 0.5, 0.2, key="h_dropout")
        train_h    = st.button("🚀 Entraîner le modèle", type="primary", key="train_house")

    with col2:
        if train_h:
            with st.spinner("Entraînement de l'ANN…"):
                try:
                    import tensorflow as tf
                    from tensorflow import keras
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import (
                        mean_squared_error, mean_absolute_error, r2_score
                    )

                    # ---- Données ----
                    if house_file:
                        df = pd.read_csv(house_file)
                        target = "price" if "price" in df.columns else df.select_dtypes(
                            include=[np.number]).columns[-1]
                        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                                    if c != target]
                        df = df[num_cols + [target]].dropna()
                        X, y = df[num_cols].values, df[target].values
                        st.success(f"Dataset chargé : {df.shape[0]} lignes, {len(num_cols)} features")
                    else:
                        np.random.seed(42)
                        n = 2000
                        bedrooms  = np.random.randint(1, 6, n)
                        bathrooms = np.random.uniform(1, 4, n)
                        sqft      = np.random.randint(500, 5000, n)
                        floors    = np.random.randint(1, 4, n)
                        yr_built  = np.random.randint(1950, 2020, n)
                        condition = np.random.randint(1, 6, n)
                        X = np.column_stack([bedrooms, bathrooms, sqft, floors, yr_built, condition])
                        y = (200_000 + sqft * 150 + bedrooms * 20_000 + bathrooms * 15_000
                             + condition * 10_000 + np.random.normal(0, 30_000, n))
                        y = np.maximum(y, 50_000)

                    sx, sy = StandardScaler(), StandardScaler()
                    Xs = sx.fit_transform(X)
                    ys = sy.fit_transform(y.reshape(-1, 1)).flatten()
                    X_tr, X_te, y_tr, y_te = train_test_split(Xs, ys, test_size=0.2, random_state=42)

                    # ---- Modèle ----
                    tf.random.set_seed(42)
                    model = keras.Sequential()
                    model.add(keras.layers.Input(shape=(X_tr.shape[1],)))
                    for u in (h_layers or [128, 64]):
                        model.add(keras.layers.Dense(u, activation="relu"))
                        model.add(keras.layers.BatchNormalization())
                        model.add(keras.layers.Dropout(h_dropout))
                    model.add(keras.layers.Dense(1))
                    model.compile(optimizer=keras.optimizers.Adam(h_lr), loss="mse", metrics=["mae"])

                    hist = model.fit(X_tr, y_tr, validation_split=0.2,
                                     epochs=h_epochs, batch_size=h_batch, verbose=0)

                    # ---- Évaluation ----
                    y_pred_s = model.predict(X_te, verbose=0).flatten()
                    y_pred   = sy.inverse_transform(y_pred_s.reshape(-1, 1)).flatten()
                    y_true   = sy.inverse_transform(y_te.reshape(-1, 1)).flatten()

                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    mae  = mean_absolute_error(y_true, y_pred)
                    r2   = r2_score(y_true, y_pred)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("RMSE", f"${rmse:,.0f}")
                    m2.metric("MAE", f"${mae:,.0f}")
                    m3.metric("R²", f"{r2:.4f}")
                    m4.metric("Test samples", len(y_te))

                    fig_l = go.Figure()
                    fig_l.add_trace(go.Scatter(y=hist.history["loss"],     name="Train Loss",
                                               line=dict(color="#667eea")))
                    fig_l.add_trace(go.Scatter(y=hist.history["val_loss"], name="Val Loss",
                                               line=dict(color="#f093fb")))
                    fig_l.update_layout(title="Courbe de perte", xaxis_title="Epoch",
                                        yaxis_title="MSE", template="plotly_dark")
                    st.plotly_chart(fig_l, use_container_width=True)

                    idx = np.random.choice(len(y_true), min(200, len(y_true)), replace=False)
                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(x=y_true[idx], y=y_pred[idx], mode="markers",
                                               name="Prédictions",
                                               marker=dict(color="#667eea", opacity=0.6)))
                    lo, hi = y_true.min(), y_true.max()
                    fig_p.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                               name="Parfait", line=dict(color="red", dash="dash")))
                    fig_p.update_layout(title="Prédictions vs Réel", xaxis_title="Prix réel ($)",
                                        yaxis_title="Prix prédit ($)", template="plotly_dark")
                    st.plotly_chart(fig_p, use_container_width=True)
                    st.success("✅ Modèle entraîné avec succès !")

                except ImportError:
                    st.error("TensorFlow non installé. Lancez : `pip install tensorflow`")
                except Exception as e:
                    st.error(f"Erreur : {e}")
        else:
            st.info("👈 Configurez les paramètres puis cliquez sur **Entraîner le modèle**.")
            if h_layers:
                st.code("Input → " + " → ".join(
                    [f"Dense({u})+BN+Dropout" for u in h_layers]) + " → Output")

# ============================================================
#  TAB 2 — DIABETES CLASSIFICATION (ANN)
# ============================================================
with tabs[1]:
    st.header("🩺 Classification du Diabète — Artificial Neural Network")
    st.markdown(
        "**Dataset :** [Diabetes Health Indicators — Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)  \n"
        "Classification binaire du diabète via un ANN TensorFlow/Keras."
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Configuration")
        diab_file = st.file_uploader("Upload diabetes CSV", type=["csv"], key="diab_data")
        if diab_file is None:
            st.info("Données synthétiques utilisées si aucun fichier n'est chargé.")

        d_epochs  = st.slider("Epochs", 10, 150, 50, key="d_epochs")
        d_batch   = st.selectbox("Batch Size", [32, 64, 128, 256], index=1, key="d_batch")
        d_layers  = st.multiselect("Hidden Layers", [32, 64, 128, 256],
                                    default=[128, 64, 32], key="d_layers")
        d_lr      = st.select_slider("Learning Rate", [0.0001, 0.001, 0.01], value=0.001, key="d_lr")
        d_cw      = st.checkbox("Compenser le déséquilibre des classes", value=True, key="d_cw")
        train_d   = st.button("🚀 Entraîner le modèle Diabète", type="primary", key="train_diab")

    with col2:
        if train_d:
            with st.spinner("Entraînement ANN classification…"):
                try:
                    import tensorflow as tf
                    from tensorflow import keras
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import (
                        classification_report, confusion_matrix,
                        roc_auc_score, roc_curve
                    )
                    from sklearn.utils.class_weight import compute_class_weight

                    if diab_file:
                        df = pd.read_csv(diab_file)
                        st.success(f"Dataset : {df.shape[0]} lignes, {df.shape[1]} colonnes")
                        if "Diabetes_binary" in df.columns:
                            target = "Diabetes_binary"
                        elif "diabetes" in df.columns:
                            target = "diabetes"
                        else:
                            target = df.columns[-1]
                        feats = [c for c in df.select_dtypes(include=[np.number]).columns
                                 if c != target]
                        df = df[feats + [target]].dropna()
                        X, y = df[feats].values, df[target].values.astype(int)
                    else:
                        np.random.seed(42)
                        n = 3000
                        X = np.column_stack([
                            np.random.normal(28, 6, n),
                            np.random.randint(1, 13, n),
                            np.random.randint(0, 2, n),
                            np.random.randint(0, 2, n),
                            np.random.randint(0, 2, n),
                            np.random.randint(0, 2, n),
                            np.random.randint(1, 6, n),
                            np.random.randint(0, 31, n),
                            np.random.randint(0, 31, n),
                            np.random.randint(0, 2, n),
                            np.random.randint(0, 2, n),
                            np.random.randint(0, 2, n),
                        ])
                        risk = (0.3 * (X[:, 0] > 30) + 0.2 * X[:, 2] + 0.15 * X[:, 3]
                                + 0.1 * (X[:, 1] > 8) - 0.1 * X[:, 4])
                        prob = 1 / (1 + np.exp(-3 * (risk - 0.3)))
                        y = (np.random.random(n) < prob).astype(int)

                    sx = StandardScaler()
                    Xs = sx.fit_transform(X)
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        Xs, y, test_size=0.2, random_state=42, stratify=y)

                    cw_dict = None
                    if d_cw:
                        cls = np.unique(y_tr)
                        cw  = compute_class_weight("balanced", classes=cls, y=y_tr)
                        cw_dict = dict(zip(cls, cw))

                    tf.random.set_seed(42)
                    model = keras.Sequential()
                    model.add(keras.layers.Input(shape=(X_tr.shape[1],)))
                    for u in (d_layers or [128, 64, 32]):
                        model.add(keras.layers.Dense(u, activation="relu"))
                        model.add(keras.layers.BatchNormalization())
                        model.add(keras.layers.Dropout(0.3))
                    model.add(keras.layers.Dense(1, activation="sigmoid"))
                    model.compile(
                        optimizer=keras.optimizers.Adam(d_lr),
                        loss="binary_crossentropy",
                        metrics=["accuracy", keras.metrics.AUC(name="auc")],
                    )

                    hist = model.fit(
                        X_tr, y_tr, validation_split=0.2,
                        epochs=d_epochs, batch_size=d_batch,
                        class_weight=cw_dict, verbose=0,
                    )

                    y_prob = model.predict(X_te, verbose=0).flatten()
                    y_pred = (y_prob > 0.5).astype(int)
                    rep    = classification_report(y_te, y_pred, output_dict=True)
                    auc    = roc_auc_score(y_te, y_prob)

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{rep['accuracy']:.4f}")
                    m2.metric("AUC-ROC",  f"{auc:.4f}")
                    m3.metric("F1 (weighted)", f"{rep['weighted avg']['f1-score']:.4f}")
                    m4.metric("Recall (weighted)", f"{rep['weighted avg']['recall']:.4f}")

                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(y=hist.history["accuracy"],
                                                 name="Train Acc", line=dict(color="#667eea")))
                    fig_acc.add_trace(go.Scatter(y=hist.history["val_accuracy"],
                                                 name="Val Acc",   line=dict(color="#f093fb")))
                    fig_acc.update_layout(title="Accuracy", xaxis_title="Epoch",
                                          yaxis_title="Accuracy", template="plotly_dark")
                    st.plotly_chart(fig_acc, use_container_width=True)

                    c1, c2 = st.columns(2)
                    with c1:
                        cm = confusion_matrix(y_te, y_pred)
                        fig_cm = px.imshow(
                            cm, text_auto=True,
                            labels=dict(x="Prédit", y="Réel", color="Count"),
                            x=["No Diabetes", "Diabetes"],
                            y=["No Diabetes", "Diabetes"],
                            title="Matrice de confusion",
                            color_continuous_scale="Blues",
                        )
                        st.plotly_chart(fig_cm, use_container_width=True)
                    with c2:
                        fpr, tpr, _ = roc_curve(y_te, y_prob)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                                                     name=f"ROC (AUC={auc:.3f})",
                                                     line=dict(color="#667eea")))
                        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Aléatoire",
                                                     line=dict(color="red", dash="dash")))
                        fig_roc.update_layout(title="Courbe ROC", xaxis_title="FPR",
                                              yaxis_title="TPR", template="plotly_dark")
                        st.plotly_chart(fig_roc, use_container_width=True)

                    st.success("✅ Modèle de classification diabète entraîné !")

                except ImportError:
                    st.error("TensorFlow non installé. Lancez : `pip install tensorflow`")
                except Exception as e:
                    st.error(f"Erreur : {e}")
        else:
            st.info("👈 Configurez les paramètres puis cliquez sur **Entraîner le modèle Diabète**.")

# ============================================================
#  TAB 3 — RAG ON PDF
# ============================================================
with tabs[2]:
    st.header("📄 RAG — Retrieval Augmented Generation sur PDF")
    st.markdown(
        "Chargez un PDF et posez des questions. Utilise **sentence-transformers**, "
        "**FAISS** et **LangChain** pour le pipeline RAG."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Document")
        rag_pdf  = st.file_uploader("Upload PDF", type=["pdf"], key="rag_pdf")
        st.subheader("🔑 LLM")
        llm_prov = st.selectbox("Provider LLM",
                                 ["Anthropic (Claude)", "Mode Offline (similarité seule)"],
                                 key="llm_prov")
        anth_key = ""
        if llm_prov == "Anthropic (Claude)":
            anth_key = st.text_input("Anthropic API Key", type="password",
                                     value=os.environ.get("ANTHROPIC_API_KEY", ""),
                                     key="rag_anth_key")
        chunk_sz  = st.slider("Taille chunk (chars)", 200, 1000, 500, key="chunk_sz")
        chunk_ov  = st.slider("Chevauchement chunk", 0, 200, 50, key="chunk_ov")
        top_k     = st.slider("Top-K résultats", 1, 10, 3, key="top_k")
        proc_btn  = st.button("📑 Traiter le document", type="primary", key="proc_pdf")

    with col2:
        st.subheader("💬 Poser une question")
        question  = st.text_area("Votre question :", height=100,
                                  placeholder="De quoi parle ce document ?", key="rag_q")
        ask_btn   = st.button("🔍 Demander", type="primary", key="ask_rag")

    if proc_btn:
        if rag_pdf is None:
            st.warning("Veuillez uploader un fichier PDF.")
        else:
            with st.spinner("Traitement du document…"):
                try:
                    import pypdf

                    reader   = pypdf.PdfReader(rag_pdf)
                    full_txt = "".join(p.extract_text() + "\n" for p in reader.pages)

                    chunks = []
                    step   = chunk_sz - chunk_ov
                    for i in range(0, len(full_txt), step):
                        chunk = full_txt[i: i + chunk_sz]
                        if len(chunk.strip()) > 50:
                            chunks.append(chunk)

                    st.session_state["pdf_chunks"] = chunks
                    st.success(f"✅ {len(reader.pages)} pages, {len(chunks)} chunks créés")

                    try:
                        from sentence_transformers import SentenceTransformer
                        import faiss

                        @st.cache_resource
                        def load_embedder():
                            return SentenceTransformer("all-MiniLM-L6-v2")

                        embedder   = load_embedder()
                        embeddings = embedder.encode(chunks, show_progress_bar=False)

                        dim   = embeddings.shape[1]
                        index = faiss.IndexFlatL2(dim)
                        index.add(embeddings.astype("float32"))

                        st.session_state["faiss_index"] = index
                        st.session_state["embedder"]    = embedder
                        st.session_state["use_faiss"]   = True
                        st.info("🔥 FAISS + Sentence Transformers activés")

                    except ImportError:
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        vec    = TfidfVectorizer(max_features=5000)
                        tfidf  = vec.fit_transform(chunks)
                        st.session_state["vectorizer"]  = vec
                        st.session_state["tfidf_matrix"] = tfidf
                        st.session_state["use_faiss"]   = False
                        st.info("TF-IDF utilisé (installez sentence-transformers + faiss-cpu pour de meilleurs résultats)")

                    with st.expander("📖 Aperçu"):
                        st.text(full_txt[:2000] + "…" if len(full_txt) > 2000 else full_txt)

                except ImportError:
                    st.error("pypdf non installé. Lancez : `pip install pypdf`")
                except Exception as e:
                    st.error(f"Erreur : {e}")

    if ask_btn and question:
        if "pdf_chunks" not in st.session_state:
            st.warning("Veuillez d'abord traiter un PDF.")
        else:
            with st.spinner("Recherche dans le document…"):
                try:
                    chunks = st.session_state["pdf_chunks"]

                    if st.session_state.get("use_faiss", False):
                        embedder = st.session_state["embedder"]
                        index    = st.session_state["faiss_index"]
                        q_emb    = embedder.encode([question])[0].astype("float32").reshape(1, -1)
                        dists, idxs = index.search(q_emb, top_k)
                        rel_chunks  = [chunks[i] for i in idxs[0] if i < len(chunks)]
                        scores      = [1 / (1 + d) for d in dists[0]]
                    else:
                        from sklearn.metrics.pairwise import cosine_similarity
                        vec   = st.session_state["vectorizer"]
                        tfidf = st.session_state["tfidf_matrix"]
                        qv    = vec.transform([question])
                        sims  = cosine_similarity(qv, tfidf)[0]
                        idxs  = sims.argsort()[-top_k:][::-1]
                        rel_chunks = [chunks[i] for i in idxs]
                        scores     = [sims[i] for i in idxs]

                    context = "\n\n".join(rel_chunks)

                    with st.expander("📋 Contexte récupéré"):
                        for i, (ch, sc) in enumerate(zip(rel_chunks, scores)):
                            st.markdown(f"**Chunk {i+1}** (score : {sc:.3f})")
                            st.text(ch[:500])
                            st.markdown("---")

                    if llm_prov == "Anthropic (Claude)" and anth_key:
                        try:
                            import anthropic
                            client = anthropic.Anthropic(api_key=anth_key)
                            prompt = (
                                f"Contexte :\n{context}\n\n"
                                f"Question : {question}\n\n"
                                "Répondez de façon concise en vous basant uniquement sur le contexte."
                            )
                            with st.spinner("Génération de la réponse avec Claude…"):
                                resp = client.messages.create(
                                    model="claude-haiku-4-5",
                                    max_tokens=1024,
                                    messages=[{"role": "user", "content": prompt}],
                                )
                            answer = resp.content[0].text
                        except Exception as e:
                            answer = f"Erreur API Claude : {e}\n\n**Extrait pertinent :**\n{rel_chunks[0][:1000]}"
                    else:
                        answer = (
                            "**Extrait le plus pertinent du document :**\n\n"
                            + rel_chunks[0]
                            + "\n\n---\n*Fournissez une clé Anthropic pour des réponses générées par IA.*"
                        )

                    st.subheader("💡 Réponse")
                    st.markdown(answer)

                except Exception as e:
                    st.error(f"Erreur : {e}")

# ============================================================
#  TAB 4 — AUDIO PROCESSING (TTS & STT)
# ============================================================
with tabs[3]:
    st.header("🎵 Traitement Audio — Text-to-Speech & Speech-to-Text")

    a_col1, a_col2 = st.columns(2)

    # ---- TTS ----
    with a_col1:
        st.subheader("🔊 Text → Speech (TTS)")
        tts_text = st.text_area(
            "Texte à convertir :",
            value="Bonjour ! Ceci est une démonstration de la synthèse vocale avec gTTS.",
            height=120, key="tts_txt",
        )
        tts_lang = st.selectbox("Langue", ["fr", "en", "es", "de", "it", "pt"], key="tts_lang")
        tts_slow = st.checkbox("Vitesse lente", value=False, key="tts_slow")

        if st.button("🎙️ Générer l'audio", type="primary", key="gen_tts"):
            if tts_text.strip():
                with st.spinner("Génération de l'audio…"):
                    try:
                        from gtts import gTTS
                        tts = gTTS(text=tts_text, lang=tts_lang, slow=tts_slow)
                        buf = io.BytesIO()
                        tts.write_to_fp(buf)
                        buf.seek(0)
                        st.success("✅ Audio généré !")
                        st.audio(buf.read(), format="audio/mp3")
                    except ImportError:
                        st.error("gTTS non installé. Lancez : `pip install gtts`")
                    except Exception as e:
                        st.error(f"Erreur TTS : {e}")
            else:
                st.warning("Entrez du texte.")

    # ---- STT ----
    with a_col2:
        st.subheader("🎤 Speech → Text (STT)")
        st.markdown("Uploadez un fichier audio pour le transcrire avec **OpenAI Whisper**.")
        aud_file   = st.file_uploader("Upload audio (MP3, WAV, M4A, OGG, FLAC)",
                                       type=["mp3", "wav", "m4a", "ogg", "flac"], key="stt_audio")
        stt_model  = st.selectbox("Taille du modèle Whisper", ["tiny", "base", "small"], key="stt_model")

        if st.button("🔍 Transcrire", type="primary", key="transcribe"):
            if aud_file is None:
                st.warning("Veuillez uploader un fichier audio.")
            else:
                with st.spinner(f"Transcription avec Whisper ({stt_model})…"):
                    try:
                        import whisper
                        suffix = "." + aud_file.name.split(".")[-1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(aud_file.read())
                            tmp_path = tmp.name
                        try:
                            wmodel = whisper.load_model(stt_model)
                            result = wmodel.transcribe(tmp_path)
                            st.success("✅ Transcription terminée !")
                            st.subheader("📝 Transcription :")
                            st.markdown(f"**{result['text']}**")
                            if result.get("segments"):
                                with st.expander("🕐 Segments horodatés"):
                                    for seg in result["segments"]:
                                        st.markdown(
                                            f"`[{seg['start']:.1f}s – {seg['end']:.1f}s]` {seg['text']}"
                                        )
                        finally:
                            os.unlink(tmp_path)
                    except ImportError:
                        st.error("openai-whisper non installé. Lancez : `pip install openai-whisper`")
                    except Exception as e:
                        st.error(f"Erreur STT : {e}")

        st.markdown("---")
        st.markdown(
            "**Formats supportés :** MP3, WAV, M4A, OGG, FLAC  \n"
            "**Modèles :** tiny (rapide), base (équilibré), small (précis)  \n"
            "**Propulsé par :** OpenAI Whisper"
        )

# ============================================================
#  TAB 5 — LANGSMITH MONITORING
# ============================================================
with tabs[4]:
    st.header("📊 Monitoring LangSmith")
    st.markdown(
        "Suivez vos runs LangChain avec **LangSmith** — latence, tokens, coûts, traces."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔑 Configuration")
        ls_key     = st.text_input("LangSmith API Key", type="password",
                                    value=os.environ.get("LANGSMITH_API_KEY", ""), key="ls_key")
        ls_project = st.text_input("Nom du projet", value="SN_ML_CLOUD_GROUPE3", key="ls_proj")
        ls_anth    = st.text_input("Anthropic API Key", type="password",
                                    value=os.environ.get("ANTHROPIC_API_KEY", ""), key="ls_anth")
        ls_query   = st.text_area("Requête à monitorer :",
                                   value="Quelles sont les étapes du machine learning supervisé ?",
                                   height=100, key="ls_query")
        run_ls     = st.button("▶️ Exécuter & Monitorer", type="primary", key="run_ls")

        st.markdown("---")
        st.subheader("📖 Configuration rapide")
        st.code("""
# Installer les dépendances
pip install langsmith langchain langchain-anthropic

# Variables d'environnement
export LANGSMITH_API_KEY="votre-clé"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_PROJECT="SN_ML_CLOUD_GROUPE3"
""", language="bash")

    with col2:
        st.subheader("📈 Résultats")
        if run_ls:
            with st.spinner("Exécution de la requête monitorée…"):
                import time as _time

                if ls_key:
                    os.environ["LANGSMITH_API_KEY"]     = ls_key
                    os.environ["LANGCHAIN_TRACING_V2"]  = "true"
                    os.environ["LANGCHAIN_PROJECT"]     = ls_project

                if ls_anth:
                    try:
                        import anthropic
                        os.environ["ANTHROPIC_API_KEY"] = ls_anth
                        client = anthropic.Anthropic(api_key=ls_anth)

                        t0   = _time.time()
                        resp = client.messages.create(
                            model="claude-haiku-4-5",
                            max_tokens=512,
                            messages=[{"role": "user", "content": ls_query}],
                        )
                        lat  = _time.time() - t0
                        ans  = resp.content[0].text

                        st.success("✅ Requête exécutée !")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Latence",        f"{lat:.2f}s")
                        m2.metric("Tokens entrée",  resp.usage.input_tokens)
                        m3.metric("Tokens sortie",  resp.usage.output_tokens)
                        st.info(ans)

                        if ls_key:
                            st.markdown(
                                "🔗 **Voir les traces :** [smith.langchain.com](https://smith.langchain.com)"
                            )
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                else:
                    st.info("Entrez les clés API pour de vraies requêtes. Données simulées :")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Latence moy.", "1.2s")
                    m2.metric("Total runs",   "42")
                    m3.metric("Taux succès",  "97.6%")

                    dates = pd.date_range(end=datetime.now(), periods=10, freq="h")
                    lats  = np.random.normal(1.2, 0.3, 10)
                    fig   = px.line(x=dates, y=lats, title="Latence dans le temps",
                                    template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.markdown("""
### 🏗️ Architecture LangSmith
```
Requête Utilisateur
    ↓
Pipeline LangChain
    ├── Prompt Template
    ├── Appel LLM (Claude)
    └── Output Parser
    ↓
LangSmith Traces
    ├── Logs entrée/sortie
    ├── Métriques de latence
    ├── Utilisation des tokens
    └── Suivi des erreurs
```
### 📊 Métriques suivies
- **Latence** : temps de réponse par run
- **Tokens** : entrée + sortie
- **Taux d'erreur** : runs échoués
- **Historique** : logs complets des conversations
""")

# ============================================================
#  TAB 6 — LSTM STOCK PREDICTION + CHATBOT
# ============================================================
with tabs[5]:
    st.header("📈 Prédiction d'Actions — LSTM + Chatbot Intelligent")

    col1, col2 = st.columns([1, 2])

    COMPANIES = {
        "Amazon (AMZN)":          "AMZN",
        "Intel (INTC)":           "INTC",
        "AMD (AMD)":              "AMD",
        "General Electric (GE)":  "GE",
    }

    with col1:
        st.subheader("⚙️ Configuration")
        sel_company  = st.selectbox("Entreprise", list(COMPANIES.keys()), key="company")
        ticker       = COMPANIES[sel_company]
        hist_months  = st.slider("Période historique (mois)", 3, 12, 6, key="hist_months")
        pred_days    = st.slider("Jours à prédire", 7, 90, 30, key="pred_days")
        lstm_units   = st.slider("Unités LSTM", 32, 256, 64, key="lstm_units")
        lstm_layers  = st.slider("Couches LSTM", 1, 3, 2, key="lstm_layers")
        lstm_epochs  = st.slider("Epochs", 10, 100, 30, key="lstm_epochs")
        seq_len      = st.slider("Longueur séquence", 10, 60, 30, key="seq_len")
        train_lstm   = st.button("📊 Télécharger & Entraîner LSTM", type="primary", key="train_lstm")

        st.markdown("---")
        st.subheader("🤖 Chatbot Boursier")
        chat_key   = st.text_input("Anthropic API Key (chatbot)", type="password",
                                    value=os.environ.get("ANTHROPIC_API_KEY", ""), key="chat_key")
        chat_input = st.text_input("Votre question :",
                                    placeholder=f"Parle-moi de {sel_company}…", key="chat_in")
        send_msg   = st.button("📨 Envoyer", key="send_chat")

    with col2:
        if train_lstm:
            with st.spinner(f"Téléchargement de {ticker} et entraînement LSTM…"):
                try:
                    import yfinance as yf
                    from sklearn.preprocessing import MinMaxScaler
                    import tensorflow as tf
                    from tensorflow import keras

                    end_dt   = datetime.now()
                    start_dt = end_dt - timedelta(days=hist_months * 30)

                    df_stock = yf.Ticker(ticker).history(start=start_dt, end=end_dt)
                    if df_stock.empty:
                        st.error(f"Aucune donnée pour {ticker}")
                        st.stop()

                    st.success(f"✅ {len(df_stock)} jours de données téléchargés pour {ticker}")

                    prices  = df_stock["Close"].values.reshape(-1, 1)
                    scaler  = MinMaxScaler()
                    ps      = scaler.fit_transform(prices)

                    X_lst, y_lst = [], []
                    for i in range(seq_len, len(ps)):
                        X_lst.append(ps[i - seq_len:i, 0])
                        y_lst.append(ps[i, 0])
                    X_a, y_a = np.array(X_lst), np.array(y_lst)
                    X_a = X_a.reshape(X_a.shape[0], X_a.shape[1], 1)

                    split    = int(len(X_a) * 0.8)
                    X_tr, X_te = X_a[:split], X_a[split:]
                    y_tr, y_te = y_a[:split], y_a[split:]

                    tf.random.set_seed(42)
                    model = keras.Sequential()
                    for i in range(lstm_layers):
                        ret_seq = (i < lstm_layers - 1)
                        cfg = dict(units=lstm_units, return_sequences=ret_seq)
                        if i == 0:
                            model.add(keras.layers.LSTM(**cfg, input_shape=(seq_len, 1)))
                        else:
                            model.add(keras.layers.LSTM(**cfg))
                        model.add(keras.layers.Dropout(0.2))
                    model.add(keras.layers.Dense(1))
                    model.compile(optimizer="adam", loss="mse")

                    hist_lstm = model.fit(
                        X_tr, y_tr, epochs=lstm_epochs, batch_size=32,
                        validation_data=(X_te, y_te), verbose=0,
                    )

                    tr_pred = scaler.inverse_transform(model.predict(X_tr, verbose=0))
                    te_pred = scaler.inverse_transform(model.predict(X_te, verbose=0))
                    actual  = scaler.inverse_transform(ps)

                    # Prédictions futures
                    cur_seq = ps[-seq_len:].copy()
                    fut_preds = []
                    for _ in range(pred_days):
                        inp  = cur_seq.reshape(1, seq_len, 1)
                        nxt  = model.predict(inp, verbose=0)[0, 0]
                        fut_preds.append(nxt)
                        cur_seq = np.roll(cur_seq, -1)
                        cur_seq[-1] = nxt
                    fut_prices = scaler.inverse_transform(np.array(fut_preds).reshape(-1, 1))
                    fut_dates  = [df_stock.index[-1] + timedelta(days=i + 1) for i in range(pred_days)]

                    # Graphe
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_stock.index, y=actual.flatten(),
                                             name="Réel", line=dict(color="#667eea")))
                    tr_dates = df_stock.index[seq_len: seq_len + len(tr_pred)]
                    fig.add_trace(go.Scatter(x=tr_dates, y=tr_pred.flatten(),
                                             name="Prédit (train)", line=dict(color="#f093fb", dash="dot")))
                    te_dates = df_stock.index[seq_len + len(tr_pred):]
                    if len(te_dates) > 0:
                        fig.add_trace(go.Scatter(x=te_dates[:len(te_pred)], y=te_pred.flatten(),
                                                 name="Prédit (test)", line=dict(color="#43e97b")))
                    fig.add_trace(go.Scatter(x=fut_dates, y=fut_prices.flatten(),
                                             name=f"Prévision {pred_days}j",
                                             line=dict(color="#ff6b6b", dash="dash")))
                    fig.add_vline(x=df_stock.index[-1], line_dash="dash",
                                  line_color="white", annotation_text="Aujourd'hui")
                    fig.update_layout(
                        title=f"{sel_company} — LSTM Stock Prediction",
                        xaxis_title="Date", yaxis_title="Prix (USD)",
                        template="plotly_dark", height=500,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    cp  = actual[-1][0]
                    fp  = fut_prices[-1][0]
                    pct = (fp - cp) / cp * 100
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Prix actuel",        f"${cp:.2f}")
                    m2.metric(f"Prévision J+{pred_days}", f"${fp:.2f}", delta=f"{pct:+.2f}%")
                    tr_rmse = np.sqrt(np.mean(
                        (tr_pred.flatten() - actual[seq_len: seq_len + len(tr_pred)].flatten()) ** 2
                    ))
                    m3.metric("RMSE train",  f"${tr_rmse:.2f}")
                    m4.metric("Epochs",      lstm_epochs)

                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(y=hist_lstm.history["loss"],
                                                  name="Train", line=dict(color="#667eea")))
                    fig_loss.add_trace(go.Scatter(y=hist_lstm.history["val_loss"],
                                                  name="Val",   line=dict(color="#f093fb")))
                    fig_loss.update_layout(title="Perte LSTM", template="plotly_dark", height=250)
                    st.plotly_chart(fig_loss, use_container_width=True)

                    st.session_state["stock_info"] = {
                        "company":      sel_company,
                        "ticker":       ticker,
                        "cur_price":    cp,
                        "forecast":     fp,
                        "change_pct":   pct,
                        "period":       hist_months,
                        "pred_days":    pred_days,
                    }

                except ImportError as e:
                    st.error(f"Bibliothèque manquante : {e}")
                    st.info("Installez : `pip install yfinance tensorflow`")
                except Exception as e:
                    st.error(f"Erreur : {e}")

        # ---- Chatbot ----
        if send_msg and chat_input:
            with st.spinner("Génération de la réponse…"):
                info = st.session_state.get("stock_info", {})
                ctx = ""
                if info:
                    ctx = (
                        f"\nDonnées boursières disponibles pour {info['company']} ({info['ticker']}) :\n"
                        f"- Prix actuel : ${info['cur_price']:.2f}\n"
                        f"- Prévision à {info['pred_days']} jours : ${info['forecast']:.2f} ({info['change_pct']:+.2f}%)\n"
                        f"- Période analysée : {info['period']} mois\n"
                    )

                if chat_key:
                    try:
                        import anthropic
                        client = anthropic.Anthropic(api_key=chat_key)
                        sys_msg = (
                            f"Tu es un analyste financier expert en marchés boursiers. "
                            f"Tu aides sur {sel_company} ({ticker}).{ctx}"
                            "Réponds en français, de manière claire et professionnelle."
                        )
                        resp = client.messages.create(
                            model="claude-haiku-4-5",
                            max_tokens=512,
                            system=sys_msg,
                            messages=[{"role": "user", "content": chat_input}],
                        )
                        bot_resp = resp.content[0].text
                    except Exception as e:
                        bot_resp = f"Erreur API : {e}"
                else:
                    COMPANY_INFO = {
                        "AMZN": "Amazon est une entreprise technologique multinationale spécialisée dans le e-commerce et le cloud (AWS).",
                        "INTC": "Intel est le plus grand fabricant mondial de processeurs (CPU) pour ordinateurs.",
                        "AMD":  "Advanced Micro Devices (AMD) conçoit des CPU, GPU et semi-conducteurs haute performance.",
                        "GE":   "General Electric est un conglomérat industriel dans l'énergie, le renouvelable et l'aviation.",
                    }
                    base = COMPANY_INFO.get(ticker, "Informations non disponibles.")
                    q    = chat_input.lower()
                    if any(w in q for w in ["prix", "price", "valeur", "cours"]):
                        bot_resp = f"{base}\n\n{ctx}" if ctx else base
                    elif any(w in q for w in ["prévis", "predict", "futur", "forecast"]):
                        bot_resp = ctx if ctx else "Entraînez d'abord le modèle LSTM."
                    else:
                        bot_resp = base + ("\n\n" + ctx if ctx else "")
                    bot_resp += "\n\n*Fournissez une clé Anthropic pour des réponses IA enrichies.*"

                st.subheader("🤖 Réponse :")
                st.info(bot_resp)

# ============================================================
#  TAB 7 — HUGGINGFACE TIME FORECASTING
# ============================================================
with tabs[6]:
    st.header("🔮 HuggingFace Time Series Forecasting")
    st.markdown(
        "Utilisez des **modèles HuggingFace pré-entraînés** pour des prédictions temporelles.  \n"
        "Support : `amazon/chronos-t5-tiny`, Prophet, et extrapolation statistique."
    )

    hf1, hf2 = st.columns([1, 2])

    with hf1:
        st.subheader("⚙️ Configuration")
        data_src = st.radio(
            "Source de données",
            ["Données synthétiques", "Upload CSV", "Données boursières"],
            key="hf_src",
        )

        hf_csv     = None
        hf_date_c  = "date"
        hf_val_c   = "value"
        hf_ticker2 = "AAPL"
        hf_months2 = 12

        if data_src == "Upload CSV":
            hf_csv    = st.file_uploader("CSV série temporelle", type=["csv"], key="hf_csv")
            hf_date_c = st.text_input("Colonne date", "date", key="hf_dc")
            hf_val_c  = st.text_input("Colonne valeur", "value", key="hf_vc")
        elif data_src == "Données boursières":
            hf_ticker2 = st.text_input("Ticker", "AAPL", key="hf_tick")
            hf_months2 = st.slider("Mois d'historique", 3, 24, 12, key="hf_months")
        else:
            hf_pat = st.selectbox(
                "Motif",
                ["Tendance + Saisonnalité", "Marche aléatoire", "Sinusoïde", "Croissance bruitée"],
                key="hf_pat",
            )
            hf_np = st.slider("Nombre de points", 50, 500, 200, key="hf_np")

        hf_model_sel = st.selectbox(
            "Modèle HuggingFace",
            [
                "amazon/chronos-t5-tiny (Recommandé)",
                "Prophet (Offline)",
                "Extrapolation statistique (Fallback)",
            ],
            key="hf_model",
        )
        hf_horizon = st.slider("Horizon de prédiction (jours)", 7, 90, 30, key="hf_hor")
        run_hf     = st.button("🚀 Lancer la prévision", type="primary", key="run_hf")

    with hf2:
        if run_hf:
            with st.spinner("Prévision en cours…"):
                try:
                    # ---- Préparer les données ----
                    ts     = None
                    dates  = None
                    has_bounds = False
                    lower_b = upper_b = None

                    if data_src == "Upload CSV" and hf_csv:
                        df_ts  = pd.read_csv(hf_csv, parse_dates=[hf_date_c])
                        df_ts  = df_ts[[hf_date_c, hf_val_c]].dropna()
                        df_ts.columns = ["date", "value"]
                        df_ts  = df_ts.sort_values("date")
                        ts     = df_ts["value"].values
                        dates  = pd.to_datetime(df_ts["date"])

                    elif data_src == "Données boursières":
                        try:
                            import yfinance as yf
                            end2   = datetime.now()
                            start2 = end2 - timedelta(days=hf_months2 * 30)
                            sd2    = yf.download(hf_ticker2, start=start2, end=end2, progress=False)
                            ts     = sd2["Close"].values.flatten()
                            dates  = sd2.index
                        except Exception as e:
                            st.warning(f"Données boursières introuvables ({e}). Données synthétiques utilisées.")

                    if ts is None:
                        n = hf_np if data_src == "Données synthétiques" else 200
                        t = np.arange(n)
                        pat = hf_pat if data_src == "Données synthétiques" else "Tendance + Saisonnalité"
                        if pat == "Tendance + Saisonnalité":
                            ts = 100 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 52) + np.random.normal(0, 5, n)
                        elif pat == "Marche aléatoire":
                            ts = np.cumsum(np.random.normal(0, 1, n)) + 100
                        elif pat == "Sinusoïde":
                            ts = 50 * np.sin(2 * np.pi * t / 30) + 100 + np.random.normal(0, 3, n)
                        else:
                            ts = 100 * np.exp(0.01 * t) + np.random.normal(0, 5, n)
                        dates = pd.date_range(end=datetime.now(), periods=n, freq="D")

                    if len(ts) < 20:
                        st.error("Données insuffisantes (minimum 20 points).")
                        st.stop()

                    # ---- Prévision ----
                    forecast   = None
                    model_used = ""

                    if "chronos" in hf_model_sel.lower():
                        try:
                            from chronos import ChronosPipeline
                            import torch

                            @st.cache_resource
                            def load_chronos():
                                return ChronosPipeline.from_pretrained(
                                    "amazon/chronos-t5-tiny",
                                    device_map="cpu",
                                    torch_dtype=torch.float32,
                                )

                            pipe    = load_chronos()
                            ctx     = torch.tensor(ts[-200:]).unsqueeze(0).float()
                            samples = pipe.predict(ctx, prediction_length=hf_horizon, num_samples=20)
                            # samples shape: (batch=1, num_samples, horizon)
                            forecast   = samples[0].mean(dim=0).numpy()
                            model_used = "Amazon Chronos T5 Tiny"
                            st.success("✅ Chronos T5 Tiny chargé depuis HuggingFace !")
                        except ImportError:
                            st.warning("chronos-forecasting non installé. Tentative avec Prophet…")
                        except Exception as e:
                            st.warning(f"Chronos indisponible ({e}). Tentative avec Prophet…")

                    if forecast is None:
                        try:
                            from prophet import Prophet

                            df_p = pd.DataFrame({
                                "ds": pd.to_datetime(dates[-len(ts):]),
                                "y":  ts,
                            })
                            m    = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                                           daily_seasonality=False)
                            m.fit(df_p)
                            future  = m.make_future_dataframe(periods=hf_horizon)
                            pfc     = m.predict(future)
                            forecast   = pfc["yhat"].values[-hf_horizon:]
                            lower_b    = pfc["yhat_lower"].values[-hf_horizon:]
                            upper_b    = pfc["yhat_upper"].values[-hf_horizon:]
                            has_bounds = True
                            model_used = "Facebook Prophet"
                        except ImportError:
                            pass
                        except Exception as e:
                            st.warning(f"Prophet indisponible ({e}). Fallback statistique.")

                    if forecast is None:
                        from sklearn.linear_model import LinearRegression
                        n_h  = len(ts)
                        Xr   = np.arange(n_h).reshape(-1, 1)
                        reg  = LinearRegression().fit(Xr, ts)
                        Xf   = np.arange(n_h, n_h + hf_horizon).reshape(-1, 1)
                        trend = reg.predict(Xf)
                        if n_h >= 14:
                            last14   = ts[-14:]
                            seasonal = np.tile(last14 - last14.mean(),
                                               hf_horizon // 14 + 2)[:hf_horizon]
                        else:
                            seasonal = np.zeros(hf_horizon)
                        forecast   = trend + seasonal
                        model_used = "Extrapolation statistique (Fallback)"

                    last_date = pd.to_datetime(dates[-1])
                    fut_dates = [last_date + timedelta(days=i + 1) for i in range(hf_horizon)]

                    # ---- Graphe ----
                    n_disp = min(200, len(ts))
                    fig    = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pd.to_datetime(dates[-n_disp:]), y=ts[-n_disp:],
                        name="Historique", line=dict(color="#667eea"),
                    ))
                    if has_bounds and lower_b is not None and upper_b is not None:
                        fig.add_trace(go.Scatter(
                            x=fut_dates + fut_dates[::-1],
                            y=list(upper_b) + list(lower_b[::-1]),
                            fill="toself", fillcolor="rgba(240,147,251,0.2)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="Intervalle de confiance",
                        ))
                    fig.add_trace(go.Scatter(
                        x=fut_dates, y=forecast,
                        name=f"Prévision {hf_horizon}j",
                        line=dict(color="#f093fb", dash="dash"),
                    ))
                    fig.add_vline(x=last_date, line_dash="dash", line_color="white")
                    fig.update_layout(
                        title=f"Prévision temporelle — {model_used}",
                        xaxis_title="Date", yaxis_title="Valeur",
                        template="plotly_dark", height=450,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Modèle utilisé",       model_used)
                    m2.metric("Dernière valeur",       f"{ts[-1]:.2f}")
                    chg = (forecast[-1] - ts[-1]) / abs(ts[-1]) * 100
                    m3.metric(f"Prévision J+{hf_horizon}", f"{forecast[-1]:.2f}",
                              delta=f"{chg:+.2f}%")

                    st.subheader("📋 Tableau de prévision (10 premiers jours)")
                    st.dataframe(
                        pd.DataFrame({"Date": fut_dates[:10],
                                      "Prévision": np.round(forecast[:10], 2)}),
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"Erreur : {e}")
                    st.info(
                        "Installez : `pip install chronos-forecasting` (Chronos) "
                        "ou `pip install prophet` (Prophet)"
                    )
        else:
            st.info("Configurez les paramètres et cliquez sur **Lancer la prévision**.")
            st.markdown("""
### 🤖 Modèles supportés

| Modèle | Type | Spécificité |
|--------|------|------------|
| **Chronos T5 Tiny** | Transformer | Pré-entraîné sur 700B tokens |
| **Prophet** | Statistique | Tendance + Saisonnalité |
| **Extrapolation** | Linéaire | Fallback universel |

### 📦 Installation
```bash
pip install chronos-forecasting   # Amazon Chronos (HuggingFace)
pip install prophet               # Facebook Prophet
pip install transformers torch    # HuggingFace Transformers
```
""")

# ======================== SIDEBAR ========================
with st.sidebar:
    st.title("🤖 GROUPE 3")
    st.markdown("**SN_ML_CLOUD_MBOUPDA_JOYCE**")
    st.markdown("---")
    st.markdown("### 📚 Fonctionnalités")
    for feat in [
        "🏠 Régression prix maisons (ANN)",
        "🩺 Classification diabète (ANN)",
        "📄 RAG sur PDF",
        "🎵 TTS & STT Audio",
        "📊 LangSmith Monitoring",
        "📈 LSTM Prédiction actions",
        "🔮 HF Time Forecasting",
    ]:
        st.markdown(f"- {feat}")

    st.markdown("---")
    st.markdown("### 🔧 Statut dépendances")
    for lib, imp in [
        ("TensorFlow", "tensorflow"),
        ("yfinance",   "yfinance"),
        ("sentence-transformers", "sentence_transformers"),
        ("faiss-cpu",  "faiss"),
        ("openai-whisper", "whisper"),
        ("gTTS",       "gtts"),
        ("pypdf",      "pypdf"),
        ("LangSmith",  "langsmith"),
        ("Chronos",    "chronos"),
        ("Prophet",    "prophet"),
    ]:
        try:
            __import__(imp)
            st.success(f"{lib} ✓")
        except ImportError:
            st.warning(f"{lib} ✗")

    st.markdown("---")
    st.caption("Développé pour ESTIAM — Cours Cloud ML 2025")
