import cv2

# Créez un objet VideoCapture pour accéder à la webcam (index 0)
cap = cv2.VideoCapture(0)

# Vérifiez si la webcam est ouverte
if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la webcam.")
    exit()

# Boucle infinie pour afficher le flux vidéo
while True:
    # Lisez une image depuis la webcam
    ret, frame = cap.read()

    # Vérifiez si la lecture est réussie
    if not ret:
        print("Erreur: Impossible de lire l'image depuis la webcam.")
        break

    # Affichez l'image dans une fenêtre
    cv2.imshow("Webcam", frame)

    # Attendez 25 ms et arrêtez la boucle si la touche 'q' est pressée
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Libérez la webcam et fermez la fenêtre
cap.release()
cv2.destroyAllWindows()
