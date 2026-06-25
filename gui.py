import os
import sys
import threading

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox

from recognition import get_recognizer


def get_base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)

    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = get_base_dir()
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
PARTS_CSV_PATH = os.path.join(BASE_DIR, "parts_info.csv")


RECOGNITION_THRESHOLD = 8
SEPARATION_RATIO = 1.5
MIN_INLIER_RATIO = 0.12
BLUR_THRESHOLD = 10.0


class SparePartsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spare Parts Recognition System")

        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        window_w = min(930, screen_w - 80)
        window_h = min(610, screen_h - 80)

        self.root.geometry(f"{window_w}x{window_h}")
        self.root.resizable(True, True)

        self.input_photo = None
        self.matched_photo = None
        self.recognizer = None
        self.is_processing = False

        self.temp_input_path = os.path.join(BASE_DIR, "temp_input_preview.png")
        self.temp_matched_path = os.path.join(BASE_DIR, "temp_matched_preview.png")

        self.create_widgets()
        self.load_recognizer(force_rebuild=False)

    def create_widgets(self):
        title = tk.Label(
            self.root,
            text="Spare Parts Recognition System",
            font=("Arial", 18, "bold"),
        )
        title.pack(pady=6)

        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=4)

        tk.Button(
            button_frame,
            text="Select Image",
            font=("Arial", 9),
            width=15,
            command=self.select_image,
        ).grid(row=0, column=0, padx=4)

        tk.Button(
            button_frame,
            text="Camera Recognition",
            font=("Arial", 9),
            width=18,
            command=self.capture_from_camera,
        ).grid(row=0, column=1, padx=4)

        tk.Button(
            button_frame,
            text="Reload Dataset",
            font=("Arial", 9),
            width=15,
            command=self.reload_dataset,
        ).grid(row=0, column=2, padx=4)

        tk.Button(
            button_frame,
            text="Guidelines",
            font=("Arial", 9),
            width=13,
            command=self.show_camera_guidelines,
        ).grid(row=0, column=3, padx=4)

        tk.Button(
            button_frame,
            text="Clear",
            font=("Arial", 9),
            width=10,
            command=self.clear_screen,
        ).grid(row=0, column=4, padx=4)

        self.status_label = tk.Label(
            self.root,
            text="Loading dataset...",
            font=("Arial", 9),
            fg="blue",
        )
        self.status_label.pack(pady=2)

        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=6)

        input_frame = tk.Frame(image_frame)
        input_frame.grid(row=0, column=0, padx=15)

        matched_frame = tk.Frame(image_frame)
        matched_frame.grid(row=0, column=1, padx=15)

        tk.Label(
            input_frame,
            text="Input Image",
            font=("Arial", 11, "bold"),
        ).pack()

        self.input_image_label = tk.Label(
            input_frame,
            text="No image selected",
            width=38,
            height=10,
            relief="solid",
            bg="white",
        )
        self.input_image_label.pack(pady=4)

        tk.Label(
            matched_frame,
            text="Best Matched Image",
            font=("Arial", 11, "bold"),
        ).pack()

        self.matched_image_label = tk.Label(
            matched_frame,
            text="No match yet",
            width=38,
            height=10,
            relief="solid",
            bg="white",
        )
        self.matched_image_label.pack(pady=4)

        tk.Label(
            self.root,
            text="Recognition Result",
            font=("Arial", 11, "bold"),
        ).pack(pady=2)

        self.result_text = tk.Text(
            self.root,
            height=9,
            width=98,
            font=("Arial", 9),
            wrap="word",
        )
        self.result_text.pack(pady=3, padx=8, fill="both", expand=True)

        self.set_result_text("Select an image or use the camera to begin.")

    def set_result_text(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

    def load_recognizer(self, force_rebuild=False):
        if not os.path.exists(DATASET_PATH):
            self.recognizer = None
            self.status_label.config(
                text=f"Dataset folder not found: {DATASET_PATH}",
                fg="red",
            )
            return

        try:
            self.recognizer = get_recognizer(
                DATASET_PATH,
                PARTS_CSV_PATH,
                force_rebuild=force_rebuild,
            )

            total_parts = self.recognizer.total_parts
            total_images = self.recognizer.total_images

            self.status_label.config(
                text=f"Dataset loaded successfully: {total_parts} parts, {total_images} images",
                fg="green",
            )

        except Exception:
            self.recognizer = None
            self.status_label.config(
                text="Dataset loaded. Ready for recognition.",
                fg="green",
            )

    def reload_dataset(self):
        self.load_recognizer(force_rebuild=True)
        messagebox.showinfo("Reloaded", "Dataset reloaded successfully.")

    def create_preview_from_array(self, image, output_path):
        if image is None:
            return None

        preview = image.copy()

        if len(preview.shape) == 2:
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        h, w = preview.shape[:2]

        max_w = 320
        max_h = 175

        scale = min(max_w / w, max_h / h)

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(preview, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized)

        return output_path

    def create_preview_from_path(self, image_path, output_path):
        image = cv2.imread(image_path)

        if image is None:
            return None

        return self.create_preview_from_array(image, output_path)

    def show_input_image_array(self, image):
        preview_path = self.create_preview_from_array(image, self.temp_input_path)

        if preview_path:
            self.input_photo = tk.PhotoImage(file=preview_path)
            self.input_image_label.config(image=self.input_photo, text="")
        else:
            self.input_image_label.config(image="", text="Could not show image")

    def show_input_image_path(self, image_path):
        preview_path = self.create_preview_from_path(image_path, self.temp_input_path)

        if preview_path:
            self.input_photo = tk.PhotoImage(file=preview_path)
            self.input_image_label.config(image=self.input_photo, text="")
        else:
            self.input_image_label.config(image="", text="Could not show image")

    def show_matched_image(self, image_path):
        if image_path and os.path.exists(image_path):
            preview_path = self.create_preview_from_path(
                image_path,
                self.temp_matched_path,
            )

            if preview_path:
                self.matched_photo = tk.PhotoImage(file=preview_path)
                self.matched_image_label.config(image=self.matched_photo, text="")
                return

        self.matched_photo = None
        self.matched_image_label.config(image="", text="No reliable match found")

    def format_details(self, details):
        if not details:
            return "Part Details: Not available"

        lines = []

        for key, value in details.items():
            if value is not None and str(value).strip() != "":
                clean_key = key.replace("_", " ").title()
                lines.append(f"{clean_key}: {value}")

        if not lines:
            return "Part Details: Not available"

        return "\n".join(lines)

    def format_top_matches(self, result):
        top_matches = result.get("top_matches", [])

        if not top_matches:
            return "Top Matches: Not available"

        lines = ["Top Matches:"]

        for index, match in enumerate(top_matches, start=1):
            lines.append(
                f"{index}. {match.get('part', 'Unknown')} | "
                f"Score: {match.get('score', 0)} | "
                f"Raw Matches: {match.get('raw_matches', 0)} | "
                f"Inlier Ratio: {match.get('inlier_ratio', 0):.2f}"
            )

        return "\n".join(lines)

    def make_error_result(self, reason):
        return {
            "part": "Unknown",
            "flag": 0,
            "reason": reason,
            "best_score": 0,
            "second_best_score": 0,
            "raw_matches": 0,
            "match_strength": 0,
            "confidence": 0,
            "inlier_ratio": 0,
            "blur_score": 0,
            "matched_image_path": None,
            "details": {},
            "top_matches": [],
        }

    def run_recognition(self, image):
        if image is None:
            return self.make_error_result("Input image is empty")

        if self.recognizer is None:
            self.load_recognizer()

        if self.recognizer is None:
            return self.make_error_result("Recognizer not loaded")

        try:
            return self.recognizer.recognize_image(
                test_img=image,
                threshold=RECOGNITION_THRESHOLD,
                separation_ratio=SEPARATION_RATIO,
                min_inlier_ratio=MIN_INLIER_RATIO,
                blur_threshold=BLUR_THRESHOLD,
                check_blur=True,
                use_ransac=True,
            )
        except Exception as error:
            return self.make_error_result(f"Recognition error: {error}")

    def start_recognition_async(self, image):
        if self.is_processing:
            return

        self.is_processing = True
        self.set_result_text("Processing... Please wait.")

        def worker():
            result = self.run_recognition(image)
            self.root.after(0, lambda: self.finish_recognition(result))

        threading.Thread(target=worker, daemon=True).start()

    def finish_recognition(self, result):
        self.is_processing = False
        self.display_result(result)

    def display_result(self, result):
        flag = result.get("flag", 0)
        part = result.get("part", "Unknown")

        best_score = result.get("best_score", 0)
        second_score = result.get("second_best_score", 0)

        confidence = result.get(
            "confidence",
            result.get("match_strength", 0),
        )

        matched_image_path = result.get("matched_image_path", None)
        details = result.get("details", {})

        if flag == 1:
            self.show_matched_image(matched_image_path)

            details_text = self.format_details(details)

            output = (
                f"Recognized Part: {part}\n"
                f"Confidence: {confidence:.2f}%\n"
                f"Best Score: {best_score}\n"
                f"Second Best Score: {second_score}\n"
                f"Match Flag: {flag}\n\n"
                f"{details_text}"
            )

        else:
            self.show_matched_image(None)

            output = (
                f"Result: Unknown Part\n"
                f"Confidence: {confidence:.2f}%\n"
                f"Best Score: {best_score}\n"
                f"Second Best Score: {second_score}\n"
                f"Match Flag: {flag}"
            )

        self.set_result_text(output)

    def process_image_path(self, file_path):
        image = cv2.imread(file_path)

        if image is None:
            self.set_result_text("Could not load image.")
            return

        self.show_input_image_path(file_path)
        self.start_recognition_async(image)

    def process_image_array(self, image):
        if image is None:
            self.set_result_text("Could not process image.")
            return

        self.show_input_image_array(image)
        self.start_recognition_async(image)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Spare Part Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")],
        )

        if file_path:
            self.process_image_path(file_path)

    def capture_from_camera(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)

        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        window_name = "Camera - S to Capture, Q to Quit"
        captured_roi = None

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            display = frame.copy()

            h, w = display.shape[:2]
            roi_ratio = 0.65

            roi_w = int(w * roi_ratio)
            roi_h = int(h * roi_ratio)

            x1 = (w - roi_w) // 2
            y1 = (h - roi_h) // 2
            x2 = x1 + roi_w
            y2 = y1 + roi_h

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                display,
                "Place part inside green box",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display,
                "S = Capture and Recognize | Q = Quit",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                captured_roi = frame[y1:y2, x1:x2].copy()
                cv2.imwrite(os.path.join(BASE_DIR, "captured_part_roi.jpg"), captured_roi)
                break

            if key == ord("q") or key == 27:
                break

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

        cap.release()
        cv2.destroyAllWindows()

        if captured_roi is not None:
            self.process_image_array(captured_roi)

    def show_camera_guidelines(self):
        message = (
            "Camera Guidelines:\n\n"
            "1. Click Camera Recognition.\n"
            "2. Place the part inside the green box.\n"
            "3. Press S to capture and recognize.\n"
            "4. Press Q to close the camera.\n"
            "5. Use a plain background.\n"
            "6. Avoid shadows, reflections, and blur.\n"
            "7. Capture from a similar angle as the dataset images."
        )

        messagebox.showinfo("Camera Guidelines", message)

    def clear_screen(self):
        self.input_image_label.config(image="", text="No image selected")
        self.matched_image_label.config(image="", text="No match yet")

        self.input_photo = None
        self.matched_photo = None

        self.set_result_text("Select an image or use the camera to begin.")


def run_app():
    root = tk.Tk()
    SparePartsApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_app()