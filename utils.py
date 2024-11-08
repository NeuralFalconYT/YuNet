import numpy as np
import cv2 as cv
def analyze_results(results,threshold=0.0):
    faces=[]
    for det in results:
        conf = det[-1]
        if conf>threshold:
            face={}
            bbox = det[0:4].astype(np.int32)
            face["bbox"]=[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            face["confidence"]=conf
            landmarks = det[4:14].astype(np.int32).reshape((5,2))
            face["landmarks"]={}
            # wnat as metion which is righ eye, left eye, nose tip, right mouth corner, left mouth corner
            face["landmarks"]["right_eye"]=tuple(landmarks[0])
            face["landmarks"]["left_eye"]=tuple(landmarks[1])
            face["landmarks"]["nose_tip"]=tuple(landmarks[2])
            face["landmarks"]["right_mouth_corner"]=tuple(landmarks[3])
            face["landmarks"]["left_mouth_corner"]=tuple(landmarks[4])
            faces.append(face)
    return faces
    # print(faces)
    # [{'bbox': [176, 196, 274, 334], 'confidence': 0.9400955, 'landmarks': {'right_eye': (188, 251), 'left_eye': (226, 251), 'nose_tip': (192, 278), 'right_mouth_corner': (193, 302), 'left_mouth_corner': (227, 301)}}]

def hex_to_bgr(hex_code):
    # Remove '#' if present
    hex_code = hex_code.lstrip('#')
    # Convert hex to BGR tuple (reversed order)
    return tuple(int(hex_code[i:i+2], 16) for i in (4, 2, 0))

# print(faces)
# [{'bbox': [176, 196, 274, 334], 'confidence': 0.9400955, 'landmarks': {'right_eye': (188, 251), 'left_eye': (226, 251), 'nose_tip': (192, 278), 'right_mouth_corner': (193, 302), 'left_mouth_corner': (227, 301)}}]

def visualize(image, faces,keypoints=False,display_prediction_labels=True):
    t=2
    for face in faces:
        bbox = face["bbox"]
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        # Set `l` to a percentage of the bounding box width or height
        l = min(width, height) // 5  # 20% of the smallest dimension

        color = hex_to_bgr("#00ff51")

        # Draw bounding box
        cv.rectangle(image, (x1, y1), (x2, y2), color, 1)
        # confidence = round(face["confidence"], 2)
        # cv.putText(image, str(confidence), (x1, y1 -10), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        # Draw top-left corner
        cv.line(image, (x1, y1), (x1 + l, y1), color, thickness=t)
        cv.line(image, (x1, y1), (x1, y1 + l), color, thickness=t)
        # Draw top-right corner
        cv.line(image, (x2, y1), (x2 - l, y1), color, thickness=t)
        cv.line(image, (x2, y1), (x2, y1 + l), color, thickness=t)
        # Draw bottom-left corner
        cv.line(image, (x1, y2), (x1 + l, y2), color, thickness=t)
        cv.line(image, (x1, y2), (x1, y2 - l), color, thickness=t)
        # Draw bottom-right corner
        cv.line(image, (x2, y2), (x2 - l, y2), color, thickness=t)
        cv.line(image, (x2, y2), (x2, y2 - l), color, thickness=t)
        if keypoints:
            # Draw landmarks with distinct colors
            landmarks = face["landmarks"]
            right_eye = landmarks["right_eye"]
            left_eye = landmarks["left_eye"]
            nose_tip = landmarks["nose_tip"]
            right_mouth_corner = landmarks["right_mouth_corner"]
            left_mouth_corner = landmarks["left_mouth_corner"]
            cv.circle(image, right_eye, 2, hex_to_bgr("#ffffff"), 2)
            cv.circle(image, left_eye, 2, hex_to_bgr("#ff0000"), 2)
            cv.circle(image, nose_tip, 2, hex_to_bgr("#00ffd8"), 2)
            cv.circle(image, right_mouth_corner, 2, hex_to_bgr("#ff00ff"), 2)
            cv.circle(image, left_mouth_corner, 2, hex_to_bgr("#ffd500"), 2)
                    # confidence = round(face["confidence"], 2)
        # cv.putText(image, str(confidence), (x1, y1 -10), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))

        if display_prediction_labels:
            margin = 5
            thickness = 1
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            text = '{:.2f}'.format(face["confidence"])
            text_size = cv.getTextSize(text, font, font_scale, thickness)[0]
            # Adjust the text position relative to the box
            text_x = bbox[0]
            text_y = bbox[1] - text_size[1] - margin  # Move text up by the margin amount
            # Draw the text background rectangle
            cv.rectangle(image, (text_x, text_y), (text_x + text_size[0], text_y + text_size[1]), hex_to_bgr("#006aff"), -1)

            # Draw the text on top of the background rectangle
            cv.putText(image, text, (text_x, text_y + text_size[1]), font, font_scale, (255,255,255), thickness)
    return image