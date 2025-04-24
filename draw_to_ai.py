import cv2
import mediapipe as mp
import numpy as np
import base64
import os
from openai import OpenAI
from datetime import datetime
import requests

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

canvas = np.ones((720, 1280, 3), dtype=np.uint8) * 255
cap = cv2.VideoCapture(0)
drawing = False
prev_pinch_state = False
last_pos = None
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
drawing_path = f"images/drawings/hand_drawing_{timestamp}.png"

# Load OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Returns true then distance between thumb and index is small
def detect_pinch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = np.linalg.norm(np.array([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]))
    return distance < 0.05

def get_finger_tip(hand_landmarks, index=mp_hands.HandLandmark.INDEX_FINGER_TIP):
    return hand_landmarks.landmark[index]

# Translate image into base64 for Dalle-3
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# Generates image description, Dalle-3 prompt, and translates that prompt into an image.
def generate_image(image_path):
    b64_image = encode_image(image_path)

    # Transcribe drawing
    print("Interpreting drawing...")
    vision_prompt = "Provide a highly detailed, objective description of the scene depicted in this hand drawing. ..."
    vision_response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": vision_prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"},
                ],
            }
        ],
    )
    description = vision_response.output[0].content[0].text

    # Generate text to image prompt for Dalle
    print("\nGenerating prompt for DALLE...")
    chat_response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": "You create detailed and visually compelling prompts for image generation."},
            {"role": "user", "content": f"""I have a hand-drawn sketch of a scene. Here's the interpretation of the sketch:

            \"\"\"{description}\"\"\"

            Using this interpretation, create a vivid, imaginative prompt suitable for DALL·E that would turn the drawing into a full, beautiful image, while keeping the visual structure inspired by the sketch."""}
        ]
    )
    dalle_prompt = chat_response.output[0].content[0].text

    print("\nGenerating image...")
    image_response = client.images.generate(model="dall-e-3", prompt=dalle_prompt)
    print("Generated image URL:", image_response.data[0].url)

    # Fetch the generated image
    image_url = image_response.data[0].url
    response = requests.get(image_url)

    if response.status_code == 200:
        # Save image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"images/ai_generated/sketch_{timestamp}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save image to local folder
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Image saved to: {output_path}")
    else:
        print("Failed to download image.")

# Loop to draw lines when index and thumb are pinched
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape
    draw_point = None

    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label

            if label == 'Left':
                is_pinching = detect_pinch(hand_landmarks)
                if is_pinching and not prev_pinch_state:
                    drawing = not drawing
                    print("Drawing mode:", drawing)
                    last_pos = None
                prev_pinch_state = is_pinching

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
                color = (0, 255, 0) if drawing else (0, 0, 255)
                cv2.circle(frame, (x1, y1), 12, color, -1)
                cv2.circle(frame, (x2, y2), 12, color, -1)

            elif label == 'Right':
                index_tip = get_finger_tip(hand_landmarks)
                x, y = int(index_tip.x * w), int(index_tip.y * h)
                draw_point = (x, y)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if drawing and draw_point:
        if last_pos:
            cv2.line(canvas, last_pos, draw_point, (0, 0, 0), thickness=5)
        last_pos = draw_point

    preview = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Drawing Preview", preview)

    key = cv2.waitKey(1)
    if key == ord('s'): # Done drawing
        cv2.imwrite(drawing_path, canvas)
        print("Drawing saved to", drawing_path)
        generate_image(drawing_path)
    elif key == ord('q'): # When drawing isnt' satisfactory
        break

cap.release()
cv2.destroyAllWindows()
