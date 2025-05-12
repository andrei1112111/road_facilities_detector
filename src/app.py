from lgbt import lgbt
import cv2


class App:
    def __init__(self, models):
        self.models = models

    def run_rtsp(self, rtsp):
        pass

    def frame_processor(self, frame):
        pass

    def run_mp4(self, source_path, output_path):
        # Открываем видео
        cap = cv2.VideoCapture(source_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Настраиваем сохранение видео в .mp4
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        # Прогресс-бар
        pbar = lgbt(total=total_frames, desc="Processing video", mode="ussr")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with all models
            y_offset = 30
            text_height = 30
            padding = 5

            for model_name, model in self.models.items():
                # Get prediction
                result = model.predict(frame)

                # Set text color based on result
                if result > 0:
                    color = (0, 0, 0)  # Black
                    status = "OK"
                elif result < 0:
                    color = (0, 0, 255)  # Red
                    status = "PROBLEM DETECTED"
                else:
                    color = (255, 0, 0)  # Blue for unknown
                    status = "FAIL"

                # Create white background for text
                text = f"{model_name}: {status}"
                (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                # Draw background rectangle
                cv2.rectangle(frame,
                              (0, y_offset - text_height),
                              (text_width + 2 * padding, y_offset + padding),
                              (255, 255, 255), -1)

                # Put text
                cv2.putText(frame, text,
                            (padding, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                y_offset += text_height + padding

            # Write processed frame
            out.write(frame)
            pbar.update(1)

        cap.release()
        out.release()
        print(f"✅ Saved result to: {output_path}")
