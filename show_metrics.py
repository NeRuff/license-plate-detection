from ultralytics import YOLO
model = YOLO('runs/detect/runs/train/license_plate/weights/best.pt')
metrics = model.val()

print('='*50)
print('РЕЗУЛЬТАТЫ ОБУЧЕНИЯ YOLO')
print('='*50)
print(f'  mAP50 (IoU=0.5):       {metrics.box.map50:.4f} = {metrics.box.map50*100:.2f}%')
print(f'  mAP50-95 (IoU=0.5:0.95): {metrics.box.map:.4f} = {metrics.box.map*100:.2f}%')
print(f'  Precision (Точность):   {metrics.box.mp:.4f} = {metrics.box.mp*100:.2f}%')
print(f'  Recall (Полнота):      {metrics.box.mr:.4f} = {metrics.box.mr*100:.2f}%')
print('='*50)

if metrics.box.map50 >= 0.8:
    print('(mAP > 0.8)')
elif metrics.box.map50 >= 0.6:
    print('(mAP > 0.6)')
else:
    print('(< 0.6)')

print('='*50)
