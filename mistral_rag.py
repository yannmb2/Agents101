#!/usr/bin/env python
"""
rag_demo_mistral.py
──────────────────
Minimal Retrieval-Augmented-Generation demo avec Mistral AI:

  • RecursiveCharacterTextSplitter (LangChain utility only)
  • PDF + TXT ingestion
  • FAISS vector search
  • Mistral embeddings + chat
  • Chat history inside the loop

Dependencies
------------
pip install mistralai==1.* faiss-cpu numpy pypdf langchain tqdm python-dotenv
"""

from __future__ import annotations
import os, json, textwrap
from pathlib import Path
from typing import List, Dict, Any

import faiss, numpy as np
from mistralai import Mistral
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader               # <─ PDF extractor
from tqdm.auto import tqdm                # optional progress bar
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────── Config ──────────────────────────────
DOCS_DIR        = Path("./Data")             # put .txt & .pdf files here
EMBED_MODEL     = "mistral-embed"
CHAT_MODEL      = "mistral-large-latest"     # ou "mistral-medium-latest"
CHUNK_SIZE      = 800                        # characters
CHUNK_OVERLAP   = 150
TOP_K           = 4

SYSTEM_PROMPT = (
    "Tu es un tuteur précis et concis. "
    "Réponds UNIQUEMENT à partir du contexte fourni. "
    "Si la réponse est manquante, dis \"Je ne sais pas.\""
)

SYSTEM_PROMPT_CHECK = (
    "Is the following answer well-supported by the provided context? Answer yes or no, and justify briefly in french"
)



api_key = os.getenv("MISTRAL_API_KEY")
assert api_key, "👉  Veuillez définir MISTRAL_API_KEY dans votre fichier .env!"

client = Mistral(api_key=api_key)

# ─────────────────────────────────────────────────────────────────

# 1️⃣  Load & split ----------------------------------------------------------
def read_pdf_text(path: Path) -> str:
    """Extrait le texte d'un fichier PDF."""
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def load_and_split() -> List[str]:
    """Charge et divise les documents en chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks: List[str] = []
    
    print(f"🔍  Recherche de fichiers dans {DOCS_DIR}")
    
    for path in DOCS_DIR.rglob("*"):
        if path.suffix.lower() == ".txt":
            print(f"📄  Traitement: {path.name}")
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == ".pdf":
            print(f"📑  Traitement: {path.name}")
            text = read_pdf_text(path)
        else:
            continue
        
        file_chunks = splitter.split_text(text)
        # Ajouter le nom du fichier à chaque chunk pour le contexte
        for chunk in file_chunks:
            chunks.append(f"[Source: {path.name}]\n{chunk}")
    
    if not chunks:
        raise RuntimeError(f"Aucun fichier .txt ou .pdf trouvé dans {DOCS_DIR}")
    
    print(f"✅  {len(chunks)} chunks créés")
    return chunks

# 2️⃣  Mistral embeddings -----------------------------------------------------
def embed(texts: List[str]) -> List[List[float]]:
    """Génère les embeddings avec Mistral."""
    try:
        # Mistral peut traiter plusieurs textes à la fois
        response = client.embeddings.create(
            model=EMBED_MODEL,
            inputs=texts
        )
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"❌  Erreur lors de la génération des embeddings: {e}")
        raise

# 3️⃣  Build (or load) FAISS --------------------------------------------------
def get_faiss_store(chunks: List[str], idx_path: str = "faiss_mistral.index") -> tuple:
    """Construit ou charge l'index FAISS."""
    meta_path = idx_path + ".meta.json"
    
    if Path(idx_path).exists() and Path(meta_path).exists():
        print("✓  Chargement de l'index vectoriel existant …")
        index = faiss.read_index(idx_path)
        stored_chunks = json.loads(Path(meta_path).read_text(encoding='utf-8'))
        return index, stored_chunks

    print("⏳  Construction de l'index vectoriel …")
    all_vectors = []
    
    # Traitement par batch pour éviter les limites de l'API
    batch_size = 32  # Mistral recommande des batches plus petits
    for i in tqdm(range(0, len(chunks), batch_size), desc="Génération embeddings"):
        batch = chunks[i:i + batch_size]
        batch_vectors = embed(batch)
        all_vectors.extend(batch_vectors)
    
    # Conversion en matrice numpy
    mat = np.asarray(all_vectors, dtype=np.float32)
    
    # Création et sauvegarde de l'index FAISS
    index = faiss.IndexFlatL2(mat.shape[1])
    index.add(mat)
    faiss.write_index(index, idx_path)
    
    # Sauvegarde des métadonnées
    Path(meta_path).write_text(json.dumps(chunks, ensure_ascii=False), encoding='utf-8')
    
    print(f"✅  Index sauvegardé avec {len(chunks)} documents")
    return index, chunks

# 4️⃣  Retrieval --------------------------------------------------------------
def retrieve(query: str, index, chunks: List[str], k: int = TOP_K) -> List[str]:
    """Récupère les chunks les plus pertinents."""
    q_vec = np.asarray(embed([query])[0], dtype=np.float32).reshape(1, -1)
    distances, idxs = index.search(q_vec, k)
    
    retrieved_chunks = [chunks[i] for i in idxs[0]]
    
    # Affichage des scores de similarité pour le debug
    print(f"📊  Scores de similarité: {distances[0]}")
    
    return retrieved_chunks

# 5️⃣  Prompt helpers ---------------------------------------------------------
def build_user_prompt(question: str, ctx_chunks: List[str]) -> str:
    """Construit le prompt utilisateur avec le contexte."""
    context_block = "\n\n".join(
        f"[Document {i+1}]\n{chunk}" for i, chunk in enumerate(ctx_chunks)
    )
    return (
        "Utilise le contexte ci-dessous pour répondre à la question.\n\n"
        f"Contexte:\n{context_block}\n\n"
        f"Question: {question}\nRéponse:"
    )

# 6️⃣  Chat loop with history -------------------------------------------------
def chat_loop(index, chunks: List[str]):
    """Boucle de chat interactive avec historique."""
    history: List[Dict[str, str]] = []  # stocke les tours précédents
    
    print("\n🎯  RAG avec Mistral AI activé!")
    print("   Tapez votre question ou Ctrl-C pour quitter")
    
    while True:
        try:
            q = input("\n💬  Question: ").strip()
            if not q:
                continue
                
        except KeyboardInterrupt:
            print("\n👋  Au revoir!")
            break

        # Récupération du contexte pertinent
        ctx = retrieve(q, index, chunks)
        user_prompt = build_user_prompt(q, ctx)

        # Construction des messages pour l'API
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        # Affichage du contexte récupéré (transparence)
        print("\n🔍  Contexte récupéré:")
        print("─" * 80)
        for i, c in enumerate(ctx, 1):
            # Troncature pour l'affichage
            display_chunk = c[:200] + "..." if len(c) > 200 else c
            print(f"[Doc {i}] {display_chunk}")
        print("─" * 80)

        try:
            # Appel à l'API Mistral
            response = client.chat.complete(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            print("🤖  Réponse:\n")
            print(textwrap.fill(answer, width=88))
            
            # Mise à jour de l'historique avec la question simple et la réponse
            history.extend([
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer},
            ])
            
            # Limitation de l'historique pour éviter des contextes trop longs
            if len(history) > 10:  # Garde les 5 derniers échanges
                history = history[-10:]
                
        except Exception as e:
            print(f"❌  Erreur lors de l'appel à Mistral: {e}")
            continue

# 6️⃣  Chat loop with history -------------------------------------------------
def chat_loop_with_check(index, chunks: List[str]):
    """Boucle de chat interactive avec nœud de vérification."""
    history: List[Dict[str, str]] = []  # stocke les tours précédents
    
    print("\n🎯  RAG avec Mistral AI activé (avec nœud de vérification)!")
    print("   Tapez votre question ou Ctrl-C pour quitter")
    
    while True:
        try:
            q = input("\n💬  Question: ").strip()
            if not q:
                continue
                
        except KeyboardInterrupt:
            print("\n👋  Au revoir!")
            break

        # Récupération du contexte pertinent
        ctx = retrieve(q, index, chunks)
        user_prompt = build_user_prompt(q, ctx)

        # Construction des messages pour l'API
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        # Affichage du contexte récupéré (transparence)
        print("\n🔍  Contexte récupéré:")
        print("─" * 80)
        for i, c in enumerate(ctx, 1):
            # Troncature pour l'affichage
            display_chunk = c[:200] + "..." if len(c) > 200 else c
            print(f"[Doc {i}] {display_chunk}")
        print("─" * 80)

        try:
            # Appel à l'API Mistral pour la réponse principale
            response = client.chat.complete(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # === NŒUD DE VÉRIFICATION ===
            context_for_check = "\n\n".join(ctx)
            verification_prompt = (
                f"Context:\n{context_for_check}\n\n"
                f"Answer to verify:\n{answer}\n\n"
                f"Is this answer well-supported by the provided context?"
            )
            
            # Appel au nœud de vérification
            response_check = client.chat.complete(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_CHECK},
                    {"role": "user", "content": verification_prompt}
                ],
                temperature=0.1,  # Plus strict pour la vérification
                max_tokens=200
            )
            
            verification_result = response_check.choices[0].message.content.lower()
            
            # Analyse du résultat de vérification
            is_reliable = "yes" in verification_result or "oui" in verification_result
            
            # Affichage avec flag de fiabilité
            if is_reliable:
                print("🤖  Réponse (✅ Vérifiée):\n")
                print(textwrap.fill(answer, width=88))
            else:
                print("🤖  Réponse (⚠️  NON FIABLE - Contexte insuffisant):\n")
                print(textwrap.fill(answer, width=88))
                print("\n❌  ATTENTION: Cette réponse pourrait ne pas être bien supportée par les documents fournis.")
            
            print(f"\n🔍  Analyse de vérification: {response_check.choices[0].message.content}")
            print()
            
            # Mise à jour de l'historique avec la question simple et la réponse
            history.extend([
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer},
            ])
            
            # Limitation de l'historique pour éviter des contextes trop longs
            if len(history) > 10:  # Garde les 5 derniers échanges
                history = history[-10:]
                
        except Exception as e:
            print(f"❌  Erreur lors de l'appel à Mistral: {e}")
            continue

# ──────────────────────────── main ───────────────────────────────
def main():
    """Fonction principale."""
    try:
        print("🚀  Démarrage du RAG avec Mistral AI")
        
        # Vérification du dossier de données
        if not DOCS_DIR.exists():
            print(f"📁  Création du dossier {DOCS_DIR}")
            DOCS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"👉  Placez vos fichiers .txt et .pdf dans {DOCS_DIR}")
            return
        
        # Chargement et traitement des documents
        chunks = load_and_split()
        
        # Construction/chargement de l'index
        index, chunks = get_faiss_store(chunks)
        
        # Lancement de la boucle de chat
        chat_loop(index, chunks)
        
    except Exception as e:
        print(f"💥  Erreur: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())