# Utilisez une image de base Python légère
FROM python:3.9-slim-buster

# Définissez le répertoire de travail dans le conteneur
WORKDIR /app

# Copiez le fichier requirements.txt et installez les dépendance
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Copiez tout le reste de votre code dans le répertoire de travail
COPY . .

# Railway injecte automatiquement la variable d'environnement PORT.
# Cependant, il est bon de définir une valeur par défaut ou de l'exposer explicitement.
ENV PORT 8080 
EXPOSE ${PORT}

# Commande pour lancer l'application avec Gunicorn
# Assurez-vous d'avoir 'gunicorn' dans votre requirements.txt
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "app:app"]