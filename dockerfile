# Utiliser une image Python comme base
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers du projet
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip

RUN pip install -r requirements.txt
# Installer le projet en mode éditable
RUN pip install -e .
# Définir la commande par défaut (facultatif, peut être remplacé dans le workflow)
CMD ["python", "-m", "unittest", "discover", "-s", "./tests"]