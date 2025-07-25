# Tempertaure-measure-using-CV
#This project captures an image of wet and dry bulb thermometer reading using open cv and extracts the reading from the image
import cv2
import numpy as np

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return None

    print("Press 's' to capture the image, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera.")
            break

        cv2.imshow("Camera Feed", frame)

        key = cv2.waitKey(1)
        if key == ord('s'): 
            image = frame
            break
        elif key == ord('q'):  
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return image

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    return edges

def find_scale_and_readings(image):
    edges = preprocess_image(image)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    thermometers = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / float(w)
        if 2 < aspect_ratio < 10:  
            thermometers.append((x, y, w, h))

    thermometers = sorted(thermometers, key=lambda t: t[0])

    readings = {}
    for i, (x, y, w, h) in enumerate(thermometers):
        roi = image[y:y+h, x:x+w]

        reading = extract_reading_from_scale(roi)
        if i == 0:
            readings['Wet Bulb'] = reading
        elif i == 1:
            readings['Dry Bulb'] = reading

    return readings

def extract_reading_from_scale(roi):
    """
    Custom logic to extract readings from the scale.
    This can involve:
    - Template matching for digit recognition
    - Histogram projection to find the position of the marker
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    marker_position = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 2 and h > 10:  
            marker_position = y
            break

    if marker_position is None:
        return None 
    scale_height = roi.shape[0]
    max_reading = 35
    min_reading = 20 
    reading1 = max_reading - (marker_position / scale_height * max_reading)
    reading2 = min_reading - (marker_position / scale_height * min_reading)

    return round(reading1,1)
    return round (reading2,1)

def main():
    image = capture_image()
    if image is None:
        print("No image captured.")
        return

    readings = find_scale_and_readings(image)
    if readings:
        print("Extracted Readings:")
        print(f"Wet Bulb: {readings.get('Wet Bulb', 'N/A')}°C")
        print(f"Dry Bulb: {readings.get('Dry Bulb', 'N/A')}°C")
    else:
        print("Could not extract readings.")

if __name__ == "__main__":
    main()

