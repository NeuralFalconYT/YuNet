import cv2 as cv
from yunet import load_model
from utils import analyze_results,visualize
# Load the model
deviceId = 0
deviceId = "video.mp4"
model = load_model(conf_threshold=0.7, nms_threshold=0.7)


# Open the camera
cap = cv.VideoCapture(deviceId)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
model.setInputSize([w, h])

tm = cv.TickMeter()

while True:
    ret, frame = cap.read()
    # Flip the frame horizontally
    frame = cv.flip(frame, 1)
    if not ret:
        print("Failed to grab frame.")
        break
    height, width = frame.shape[:2]
    # Inference
    tm.start()
    results = model.infer(frame)
    # faces = analyze_results(results,threshold=0.5)
    faces=analyze_results(results)
    tm.stop()
    # print(faces)
    # Draw results on the frame
    if faces:
        frame = visualize(frame, faces,bounding_box=True,keypoints=False,display_prediction_labels=True,square_blur_face=True)
    # display the fps at top left corner
    cv.putText(frame, "FPS: {:.2f}".format(tm.getFPS()), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    # if height > 720:
    #     frame = cv.resize(frame, (1280, 720))
    cv.imshow('Camera Feed', frame)
    tm.reset()
    # Exit when the user presses the Esc key
    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
