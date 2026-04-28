import cv2
from pyzbar.pyzbar import decode

# new code for visual feature extraction
import numpy as np
from visual_extraction import extract_features
training_data = []
# end of new code

# Open camera
# camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
camera = cv2.VideoCapture(0) # added for camera to work on my (Szymon's) Windows laptop

# Store already seen QR codes
seen_qr = set()

# Simple phishing keywords
suspicious_keywords = ["login", "verify", "bank", "secure", "account"]

def is_suspicious(url):
    return any(word in url.lower() for word in suspicious_keywords)

if not camera.isOpened():
    print("Camera not accessible")
    exit()

while True:
    success, frame = camera.read()

    if not success:
        print("Failed to grab frame")
        break

    for qr in decode(frame):
        data = qr.data.decode('utf-8')
        x, y, w, h = qr.rect

        # Avoid duplicate spam
        if data not in seen_qr:
            seen_qr.add(data)

# ----- start of visual feature extraction -----
            cropped_qr = frame[y:y+h, x:x+w]
            visual_features = extract_features(cropped_qr)
            training_data.append({"url": data, "vgg19_features": visual_features})
# ----- end of visual feature extraction -----

            if data.startswith("http"):
                if is_suspicious(data):
                    print("POSSIBLE PHISHING:", data)
                else:
                    print("Safe URL:", data)
            else:
                print("QR says:", data)

        # Draw box
        color = (0, 255, 0)
        label = "QR"

        if data.startswith("http"):
            if is_suspicious(data):
                color = (0, 0, 255)  # red
                label = "MALICIOUS"
            else:
                color = (0, 255, 0)  # green
                label = "SAFE URL"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("QR Scanner", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):

        # added to save the training data to a file when quitting
        if len(training_data) > 0:
            np.save("qr_training_dataset.npy", training_data)
    
        break

camera.release()
cv2.destroyAllWindows()