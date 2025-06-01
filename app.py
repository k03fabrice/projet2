from flask import Flask, render_template, request, url_for
from yolo_track import ObjectDetectorAndTracker
import os
import werkzeug.utils

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Créer les dossiers s'ils n'existent pas (sécurité)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    metrics = None
    annotated_video_url = None
    error_message = None

    if request.method == 'POST':
        file = request.files.get('video')
        selected_class = request.form.get('target_class')

        if file and selected_class:
            # Sécuriser le nom du fichier uploadé pour éviter les problèmes de chemin
            video_filename = werkzeug.utils.secure_filename(file.filename)

            # Chemin complet pour enregistrer la vidéo uploadée
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)

            # Sauvegarder le fichier uploadé
            file.save(filepath)

            # Générer un nom unique pour la vidéo annotée pour éviter écrasement
            base_name = video_filename.rsplit('.', 1)[0]
            output_video_name = f"annotated_{base_name}.mp4"
            output_video_path = os.path.join(app.config['RESULTS_FOLDER'], output_video_name)

            try:
                # Initialiser et lancer la détection + tracking
                detector = ObjectDetectorAndTracker(output_video=output_video_path)
                metrics = detector.run(filepath, object_class=selected_class)

                # Générer l'URL accessible dans le template (dossier static)
                annotated_video_url = url_for('static', filename=f'results/{output_video_name}')
            except Exception as e:
                error_message = f"Erreur durant le traitement : {e}"
        else:
            error_message = "Veuillez uploader une vidéo et sélectionner une classe d'objet."

    return render_template('index.html', metrics=metrics, video_url=annotated_video_url, error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
