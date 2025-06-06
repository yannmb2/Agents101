from mistral_rag import chat_loop_with_check,load_and_split,get_faiss_store
from pathlib import Path

print("🚀  Démarrage du RAG avec Mistral AI")
DOCS_DIR = Path("./Data")
# Vérification du dossier de données
if not DOCS_DIR.exists():
    print(f"📁  Création du dossier {DOCS_DIR}")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"👉  Placez vos fichiers .txt et .pdf dans {DOCS_DIR}")
    raise 

# Chargement et traitement des documents
chunks = load_and_split()

# Construction/chargement de l'index
index, chunks = get_faiss_store(chunks)

# Lancement de la boucle de chat
chat_loop_with_check(index, chunks)