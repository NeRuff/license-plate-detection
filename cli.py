import typer
import cv2
from pathlib import Path
from loguru import logger
from model_impl import My_LicensePlate_Model
import sys
import pandas as pd

logger.remove()
logger.add("data/log_file.log", rotation="100 MB", level="INFO", format="{time} | {level} | {message}")
logger.add(sys.stdout, level="INFO")

app = typer.Typer()

def setup_logging():
    Path("data").mkdir(exist_ok=True)

@app.command()
def video(input_path: str = typer.Option(..., "--video", "-v"), output_path: str = typer.Option("./output.mp4", "--output", "-o")):
    setup_logging()
    logger.info(f"Processing video: {input_path}")
    if not Path(input_path).exists():
        logger.error(f"File not found: {input_path}")
        return
    model = My_LicensePlate_Model()
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_count = 0
    total_plates = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        plates = model.detect_plates(frame)
        total_plates += len(plates)
        frame = model.draw_boxes(frame, plates)
        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            logger.info(f"Processed frames: {frame_count}")
    cap.release()
    out.release()
    logger.success(f"Done! {frame_count} frames, {total_plates} plates found")
    logger.success(f"Result saved to {output_path}")

@app.command()
def camera(camera_id: int = typer.Option(0, "--camera", "-c")):
    setup_logging()
    logger.info(f"Starting camera {camera_id}")
    model = My_LicensePlate_Model()
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    logger.info("Press 'q' to exit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        plates = model.detect_plates(frame)
        frame = model.draw_boxes(frame, plates)
        cv2.imshow("License Plate Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Stream stopped")

@app.command()
def video_with_ocr(input_path: str = typer.Option(..., "--video", "-v"), output_path: str = typer.Option("./output_ocr.mp4", "--output", "-o")):
    setup_logging()
    logger.info(f"Processing video with OCR: {input_path}")
    if not Path(input_path).exists():
        logger.error(f"File not found: {input_path}")
        return
    model = My_LicensePlate_Model()
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    detected_plates = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        plates = model.detect_plates(frame)
        for plate in plates:
            if plate.get("text"):
                detected_plates.append({'frame': frame_count, 'plate': plate["text"], 'confidence': plate["confidence"]})
        frame = model.draw_boxes(frame, plates)
        out.write(frame)
        frame_count += 1
        if frame_count % 100 == 0:
            logger.info(f"Processed frames: {frame_count}")
    cap.release()
    out.release()
    if detected_plates:
        df = pd.DataFrame(detected_plates)
        df.to_csv("data/detected_plates.csv", index=False)
        logger.success(f"Saved {len(detected_plates)} plates to data/detected_plates.csv")
    logger.success(f"Result saved to {output_path}")

if __name__ == "__main__":
    app()
