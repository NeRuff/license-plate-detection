import cv2
from model_impl import My_LicensePlate_Model
model = My_LicensePlate_Model()
cap = cv2.VideoCapture('test_video.mp4')
plates_history = []
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    plates = model.detect_plates(frame)
    frame_count += 1
    for plate in plates:
        x1, y1, x2, y2 = plate['bbox']
        plates_history.append({
            'frame': frame_count,
            'bbox': [x1, y1, x2, y2],
            'conf': plate['confidence']
        })
cap.release()
print(f'Всего кадров: {frame_count}')
print(f'Всего детекций: {len(plates_history)}')
print(f'В среднем на кадр: {len(plates_history)/frame_count:.1f}')
