import os
import csv
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


ORB_NFEATURES = 800
MAX_IMAGE_SIDE = 650

DEFAULT_THRESHOLD = 8
DEFAULT_SEPARATION_RATIO = 1.5
DEFAULT_MIN_INLIER_RATIO = 0.12
DEFAULT_BLUR_THRESHOLD = 10.0
DEFAULT_TOP_RANSAC_CANDIDATES = 3
DEFAULT_RATIO_TEST = 0.75
DEFAULT_RANSAC_REPROJ_THRESHOLD = 5.0


def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def load_part_details(csv_path):
    details = {}

    if not csv_path or not os.path.exists(csv_path):
        return details

    try:
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames:
                return details

            possible_keys = {
                "part_id",
                "part",
                "folder",
                "part_code",
                "part_no",
                "part_number",
                "part_name",
                "part name",
                "name",
            }

            key_field = None

            for field in reader.fieldnames:
                if field.lower().strip() in possible_keys:
                    key_field = field
                    break

            if key_field is None:
                key_field = reader.fieldnames[0]

            for row in reader:
                key = row.get(key_field, "").strip()

                if key:
                    details[key] = row

    except Exception:
        return {}

    return details


def resize_keep_aspect(gray_img, max_side=MAX_IMAGE_SIDE):
    h, w = gray_img.shape[:2]
    largest = max(h, w)

    if largest <= max_side:
        return gray_img

    scale = max_side / largest
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_gray(image):
    if image is None:
        return None

    if len(image.shape) == 2:
        return image.copy()

    if len(image.shape) == 3:
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return None


def calculate_blur_score(gray_img):
    if gray_img is None:
        return 0.0

    small = resize_keep_aspect(gray_img, max_side=500)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def normalize_lighting(gray_img):
    gray_img = resize_keep_aspect(gray_img, max_side=MAX_IMAGE_SIDE)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray_img)


def prepare_image(image):
    gray = to_gray(image)

    if gray is None:
        return None, 0.0

    blur_score = calculate_blur_score(gray)
    normalized = normalize_lighting(gray)

    return normalized, blur_score


def calculate_good_matches(bf, des1, des2):
    if des1 is None or des2 is None:
        return []

    if len(des1) == 0 or len(des2) == 0:
        return []

    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return []

    good = []

    for pair in matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < DEFAULT_RATIO_TEST * n.distance:
            good.append(m)

    return good


def ransac_verify(kp1, kp2, good_matches):
    if len(good_matches) < 4:
        return 0, 0.0

    src_pts = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    try:
        _, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            DEFAULT_RANSAC_REPROJ_THRESHOLD
        )
    except cv2.error:
        return 0, 0.0

    if mask is None:
        return 0, 0.0

    inliers = int(mask.ravel().sum())
    ratio = inliers / len(good_matches)

    return inliers, ratio


@dataclass
class FeatureRecord:
    part: str
    image_path: str
    keypoints: Any
    descriptors: Any


class SparePartRecognizer:
    def __init__(self, dataset_path, csv_path=None):
        self.dataset_path = os.path.abspath(dataset_path)
        self.csv_path = os.path.abspath(csv_path) if csv_path else None

        self.orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.records = []
        self.part_details = {}
        self.load_errors = []
        self.build_time_seconds = 0.0

        self.build_index()
        self.part_details = load_part_details(self.csv_path)

    @property
    def total_images(self):
        return len(self.records)

    @property
    def total_parts(self):
        return len(set(record.part for record in self.records))

    def build_index(self):
        start = time.time()

        self.records = []
        self.load_errors = []

        if not os.path.exists(self.dataset_path):
            self.load_errors.append(f"Dataset path not found: {self.dataset_path}")
            return

        for folder in sorted(os.listdir(self.dataset_path)):
            folder_path = os.path.join(self.dataset_path, folder)

            if not os.path.isdir(folder_path):
                continue

            for filename in sorted(os.listdir(folder_path)):
                if not is_image_file(filename):
                    continue

                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)

                if image is None:
                    self.load_errors.append(f"Could not read image: {image_path}")
                    continue

                prepared, _ = prepare_image(image)

                if prepared is None:
                    self.load_errors.append(f"Could not prepare image: {image_path}")
                    continue

                kp, des = self.orb.detectAndCompute(prepared, None)

                if des is None or len(kp) == 0:
                    self.load_errors.append(f"No features found: {image_path}")
                    continue

                self.records.append(
                    FeatureRecord(
                        part=folder,
                        image_path=os.path.abspath(image_path),
                        keypoints=kp,
                        descriptors=des,
                    )
                )

        self.build_time_seconds = time.time() - start

    def reload(self):
        self.build_index()
        self.part_details = load_part_details(self.csv_path)

    def empty_result(self, part, reason, blur_score=0.0):
        return {
            "part": part,
            "best_score": 0,
            "second_best_score": 0,
            "raw_matches": 0,
            "inlier_ratio": 0.0,
            "flag": 0,
            "match_strength": 0.0,
            "confidence": 0.0,
            "matched_image_path": None,
            "details": {},
            "reason": reason,
            "blur_score": blur_score,
            "top_matches": [],
        }

    def recognize_image(
        self,
        test_img,
        threshold=DEFAULT_THRESHOLD,
        separation_ratio=DEFAULT_SEPARATION_RATIO,
        min_inlier_ratio=DEFAULT_MIN_INLIER_RATIO,
        blur_threshold=DEFAULT_BLUR_THRESHOLD,
        check_blur=True,
        use_ransac=True,
        skip_image_path=None,
    ):
        if test_img is None:
            return self.empty_result("Invalid Image", "Input image could not be loaded")

        if self.total_images == 0:
            return self.empty_result("Unknown", "Dataset has no usable feature images")

        prepared_test, blur_score = prepare_image(test_img)

        if prepared_test is None:
            return self.empty_result("Invalid Image", "Could not process input image")

        if check_blur and blur_score < blur_threshold:
            return self.empty_result(
                "Unknown",
                f"Image is blurry. Blur score {blur_score:.2f}, required {blur_threshold:.2f}",
                blur_score
            )

        kp1, des1 = self.orb.detectAndCompute(prepared_test, None)

        if des1 is None or len(kp1) == 0:
            return self.empty_result(
                "Unknown",
                "No ORB features found in input image",
                blur_score
            )

        skip_abs = os.path.abspath(skip_image_path) if skip_image_path else None
        candidates = []

        for record in self.records:
            if skip_abs and os.path.abspath(record.image_path) == skip_abs:
                continue

            good_matches = calculate_good_matches(
                self.bf,
                des1,
                record.descriptors
            )

            raw_score = len(good_matches)

            if raw_score == 0:
                continue

            candidates.append(
                {
                    "part": record.part,
                    "image_path": record.image_path,
                    "record": record,
                    "good_matches": good_matches,
                    "raw_matches": raw_score,
                    "score": raw_score,
                    "inlier_ratio": 0.0,
                }
            )

        if not candidates:
            return self.empty_result(
                "Unknown",
                "No feature matches found with dataset",
                blur_score
            )

        candidates = sorted(
            candidates,
            key=lambda x: x["raw_matches"],
            reverse=True
        )

        if use_ransac:
            for candidate in candidates[:DEFAULT_TOP_RANSAC_CANDIDATES]:
                record = candidate["record"]

                inliers, ratio = ransac_verify(
                    kp1,
                    record.keypoints,
                    candidate["good_matches"]
                )

                candidate["ransac_inliers"] = inliers
                candidate["inlier_ratio"] = ratio

                if inliers > 0:
                    candidate["score"] = inliers
                else:
                    candidate["score"] = candidate["raw_matches"]

        best_per_part = {}

        for candidate in candidates:
            part = candidate["part"]

            if part not in best_per_part:
                best_per_part[part] = candidate
            else:
                old = best_per_part[part]

                if (candidate["score"], candidate["raw_matches"]) > (
                    old["score"],
                    old["raw_matches"],
                ):
                    best_per_part[part] = candidate

        part_candidates = sorted(
            best_per_part.values(),
            key=lambda x: (x["score"], x["raw_matches"]),
            reverse=True
        )

        if not part_candidates:
            return self.empty_result(
                "Unknown",
                "No reliable part candidates found",
                blur_score
            )

        best = part_candidates[0]
        second = part_candidates[1] if len(part_candidates) > 1 else None

        best_part = best["part"]
        best_score = int(best["score"])
        second_score = int(second["score"]) if second else 0
        raw_matches = int(best["raw_matches"])
        inlier_ratio = float(best["inlier_ratio"])

        total = best_score + second_score
        match_strength = (best_score / total) * 100 if total > 0 else 0.0

        top_matches = []

        for item in part_candidates[:3]:
            top_matches.append(
                {
                    "part": item["part"],
                    "score": int(item["score"]),
                    "raw_matches": int(item["raw_matches"]),
                    "inlier_ratio": float(item["inlier_ratio"]),
                    "image_path": item["image_path"],
                }
            )

        base_result = {
            "best_score": best_score,
            "second_best_score": second_score,
            "raw_matches": raw_matches,
            "inlier_ratio": inlier_ratio,
            "match_strength": match_strength,
            "confidence": match_strength,
            "matched_image_path": best["image_path"],
            "blur_score": blur_score,
            "top_matches": top_matches,
        }

        if best_score < threshold:
            return {
                "part": "Unknown",
                "flag": 0,
                "details": {},
                "reason": f"Best score is below threshold. Best score: {best_score}, required: {threshold}",
                **base_result,
            }

        if second_score > 0 and best_score <= second_score * separation_ratio:
            return {
                "part": "Unknown",
                "flag": 0,
                "details": {},
                "reason": f"Best match is too close to second-best match. Best: {best_score}, second: {second_score}",
                **base_result,
            }

        return {
            "part": best_part,
            "flag": 1,
            "details": self.part_details.get(best_part, {}),
            "reason": "Accepted",
            **base_result,
        }


_RECOGNIZER_CACHE = {}


def get_recognizer(dataset_path, csv_path=None, force_rebuild=False):
    key = (
        os.path.abspath(dataset_path),
        os.path.abspath(csv_path) if csv_path else None,
    )

    if force_rebuild or key not in _RECOGNIZER_CACHE:
        _RECOGNIZER_CACHE[key] = SparePartRecognizer(dataset_path, csv_path)

    return _RECOGNIZER_CACHE[key]


def recognize(
    test_img,
    dataset_path,
    csv_path=None,
    threshold=DEFAULT_THRESHOLD,
    separation_ratio=DEFAULT_SEPARATION_RATIO,
    min_inlier_ratio=DEFAULT_MIN_INLIER_RATIO,
    blur_threshold=DEFAULT_BLUR_THRESHOLD,
    check_blur=True,
    use_ransac=True,
):
    recognizer = get_recognizer(dataset_path, csv_path)

    return recognizer.recognize_image(
        test_img=test_img,
        threshold=threshold,
        separation_ratio=separation_ratio,
        min_inlier_ratio=min_inlier_ratio,
        blur_threshold=blur_threshold,
        check_blur=check_blur,
        use_ransac=use_ransac,
    )
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, "dataset")
    csv_path = os.path.join(base_dir, "parts_info.csv")

    print("Loading dataset...")
    recognizer = get_recognizer(dataset_path, csv_path, force_rebuild=True)

    print("Dataset path:", dataset_path)
    print("CSV path:", csv_path)
    print("Parts found:", recognizer.total_parts)
    print("Images indexed:", recognizer.total_images)

    image_path = input("Enter test image path: ").strip().replace('"', "")

    test_img = cv2.imread(image_path)

    if test_img is None:
        print("Could not read image.")
    else:
        result = recognize(test_img, dataset_path, csv_path)

        print("\nResult:", result["part"])
        print("Flag:", result["flag"])
        print("Reason:", result["reason"])
        print("Best Score:", result["best_score"])
        print("Second Best Score:", result["second_best_score"])
        print("Raw Matches:", result["raw_matches"])
        print("Match Strength:", round(result["match_strength"], 2), "%")
        print("Blur Score:", round(result["blur_score"], 2))
        print("Matched Image:", result["matched_image_path"])
        print("Details:", result["details"])