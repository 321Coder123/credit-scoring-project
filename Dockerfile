# 1. On part d'une image Python officielle légère (slim)
FROM python:3.13-slim

# 2. On définit le dossier de travail dans le conteneur
WORKDIR /app

# 3. On copie d'abord les requirements
COPY requirements.txt .

# 4. On installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. On copie tout le reste du code (src, api, models)
COPY . .

# 6. On expose le port 8000 (celui de FastAPI)
EXPOSE 8000

# 7. La commande de démarrage
# --host 0.0.0.0 est OBLIGATOIRE dans Docker pour être accessible de l'extérieur
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]