import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100
cap = cv2.VideoCapture(0)  # Try changing to 0 or 1 depending on your system

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    class_input = input("Enter the class number (or type 'exit' to quit): ")
    
    if class_input.lower() == 'exit':
        break

    try:
        class_number = int(class_input)
    except ValueError:
        print("Invalid input. Please enter a valid class number or 'exit'.")
        continue

    class_dir = os.path.join(DATA_DIR, str(class_number))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_number}')

    done = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        cv2.putText(frame, 'Ready? Press "Q" to start! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()
