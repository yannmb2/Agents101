#!/usr/bin/env python
"""
streamlit_app.py
────────────────
Interface Streamlit pour le système RAG avec Mistral AI
"""

import streamlit as st
import os
import textwrap
from pathlib import Path
from typing import List, Dict, Any
from mistral_rag import load_and_split, get_faiss_store, retrieve, build_user_prompt, embed
from mistralai import Mistral
from dotenv import load_dotenv
import numpy as np

# Chargement des variables d'environnement
load_dotenv()

# Configuration
DOCS_DIR = Path("./Data")
CHAT_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = (
    "Tu es un tuteur précis et concis. "
    "Réponds UNIQUEMENT à partir du contexte fourni. "
    "Si la réponse est manquante, dis \"Je ne sais pas.\""
)

SYSTEM_PROMPT_CHECK = (
    "Is the following answer well-supported by the provided context? Answer yes or no, and justify briefly in french"
)

def init_mistral_client():
    """Initialise le client Mistral."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        st.error("🔑 Veuillez définir MISTRAL_API_KEY dans votre fichier .env!")
        st.stop()
    return Mistral(api_key=api_key)

def load_rag_system():
    """Charge le système RAG."""
    try:
        # Vérification du dossier de données
        if not DOCS_DIR.exists():
            st.error(f"📁 Le dossier {DOCS_DIR} n'existe pas. Créez-le et ajoutez vos documents.")
            st.stop()
        
        # Chargement avec cache
        with st.spinner("🔄 Chargement des documents..."):
            chunks = load_and_split()
        
        with st.spinner("🔄 Construction/chargement de l'index vectoriel..."):
            index, chunks = get_faiss_store(chunks)
        
        return index, chunks
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        st.stop()

def get_answer_with_verification(client, question: str, context: List[str], history: List[Dict[str, str]]):
    """Génère une réponse avec vérification."""
    user_prompt = build_user_prompt(question, context)
    
    # Construction des messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    
    # Réponse principale
    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1000
    )
    
    answer = response.choices[0].message.content
    
    # Vérification
    context_for_check = "\n\n".join(context)
    verification_prompt = (
        f"Context:\n{context_for_check}\n\n"
        f"Answer to verify:\n{answer}\n\n"
        f"Is this answer well-supported by the provided context?"
    )
    
    response_check = client.chat.complete(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_CHECK},
            {"role": "user", "content": verification_prompt}
        ],
        temperature=0.1,
        max_tokens=200
    )
    
    verification_result = response_check.choices[0].message.content
    is_reliable = "yes" in verification_result.lower() or "oui" in verification_result.lower()
    
    return answer, is_reliable, verification_result

def get_answer_simple(client, question: str, context: List[str], history: List[Dict[str, str]]):
    """Génère une réponse simple sans vérification."""
    user_prompt = build_user_prompt(question, context)
    
    # Construction des messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    
    # Réponse
    response = client.chat.complete(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def main():
    st.set_page_config(
        page_title="RAG avec Mistral AI",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG avec Mistral AI")
    st.markdown("*Système de Question-Réponse basé sur vos documents*")
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Choix du mode
        mode = st.radio(
            "Mode de fonctionnement:",
            ["Avec vérification", "Simple"],
            help="Le mode 'Avec vérification' ajoute une étape de validation de la réponse"
        )
        
        st.markdown("---")
        
        # Informations sur les documents
        if DOCS_DIR.exists():
            files = list(DOCS_DIR.glob("*.txt")) + list(DOCS_DIR.glob("*.pdf"))
            st.markdown(f"📁 **Documents trouvés:** {len(files)}")
            for file in files:
                st.markdown(f"• {file.name}")
        else:
            st.error(f"📁 Dossier {DOCS_DIR} introuvable")
        
        st.markdown("---")
        st.markdown("### 💡 Conseils d'utilisation")
        st.markdown("""
        - Posez des questions précises
        - Le système ne répond qu'à partir de vos documents
        - L'historique est conservé dans la session
        """)
    
    # Initialisation du système
    if 'rag_initialized' not in st.session_state:
        try:
            client = init_mistral_client()
            index, chunks = load_rag_system()
            st.session_state.rag_initialized = True
            st.session_state.client = client
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.success("✅ Système RAG initialisé avec succès!")
        except Exception as e:
            st.error(f"❌ Erreur d'initialisation: {e}")
            st.stop()
    
    # Initialisation de l'historique
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Interface de chat
    st.markdown("## 💬 Chat")
    
    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Affichage de la réponse avec indicateurs de fiabilité
                if "reliable" in message:
                    if message["reliable"]:
                        st.markdown("✅ **Réponse vérifiée**")
                    else:
                        st.markdown("⚠️ **Attention: Fiabilité incertaine**")
                
                st.markdown(message["content"])
                
                # Affichage du contexte utilisé
                if "context" in message:
                    with st.expander("🔍 Contexte utilisé"):
                        for i, ctx in enumerate(message["context"], 1):
                            st.markdown(f"**Document {i}:**")
                            st.text(ctx[:300] + "..." if len(ctx) > 300 else ctx)
                
                # Affichage de l'analyse de vérification
                if "verification" in message:
                    with st.expander("🔍 Analyse de vérification"):
                        st.markdown(message["verification"])
            else:
                st.markdown(message["content"])
    
    # Input pour nouvelle question
    if prompt := st.chat_input("Posez votre question..."):
        # Affichage de la question
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Traitement de la réponse
        with st.chat_message("assistant"):
            with st.spinner("🤔 Recherche dans les documents..."):
                try:
                    # Récupération du contexte
                    context = retrieve(prompt, st.session_state.index, st.session_state.chunks)
                    
                    if mode == "Avec vérification":
                        # Mode avec vérification
                        answer, is_reliable, verification = get_answer_with_verification(
                            st.session_state.client, prompt, context, st.session_state.history
                        )
                        
                        # Affichage avec indicateur de fiabilité
                        if is_reliable:
                            st.markdown("✅ **Réponse vérifiée**")
                        else:
                            st.markdown("⚠️ **Attention: Fiabilité incertaine**")
                        
                        st.markdown(answer)
                        
                        # Sauvegarde du message avec métadonnées
                        message_data = {
                            "role": "assistant",
                            "content": answer,
                            "reliable": is_reliable,
                            "verification": verification,
                            "context": context
                        }
                        
                    else:
                        # Mode simple
                        answer = get_answer_simple(
                            st.session_state.client, prompt, context, st.session_state.history
                        )
                        
                        st.markdown(answer)
                        
                        # Sauvegarde du message
                        message_data = {
                            "role": "assistant",
                            "content": answer,
                            "context": context
                        }
                    
                    # Affichage du contexte
                    with st.expander("🔍 Contexte utilisé"):
                        for i, ctx in enumerate(context, 1):
                            st.markdown(f"**Document {i}:**")
                            st.text(ctx[:300] + "..." if len(ctx) > 300 else ctx)
                    
                    # Affichage de l'analyse de vérification si applicable
                    if mode == "Avec vérification":
                        with st.expander("🔍 Analyse de vérification"):
                            st.markdown(verification)
                    
                    # Mise à jour de l'historique
                    st.session_state.messages.append(message_data)
                    st.session_state.history.extend([
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer}
                    ])
                    
                    # Limitation de l'historique
                    if len(st.session_state.history) > 10:
                        st.session_state.history = st.session_state.history[-10:]
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement: {e}")
    
    # Bouton pour effacer l'historique
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

if __name__ == "__main__":
    main()