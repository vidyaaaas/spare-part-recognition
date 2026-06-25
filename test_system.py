import os
import csv
import time

import cv2

from recognition import get_recognizer, is_image_file


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
PARTS_CSV_PATH = os.path.join(BASE_DIR, "parts_info.csv")
UNKNOWN_TESTS_PATH = os.path.join(BASE_DIR, "unknown_tests")
REPORT_PATH = os.path.join(BASE_DIR, "recognition_test_report.csv")


RECOGNITION_THRESHOLD = 15
SEPARATION_RATIO = 1.8
MIN_INLIER_RATIO = 0.20
BLUR_THRESHOLD = 35.0


def collect_dataset_images(dataset_path):
    items = []

    if not os.path.exists(dataset_path):
        return items

    for part_folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, part_folder)

        if not os.path.isdir(folder_path):
            continue

        for filename in sorted(os.listdir(folder_path)):
            if not is_image_file(filename):
                continue

            image_path = os.path.join(folder_path, filename)
            items.append(
                {
                    "image_path": image_path,
                    "expected_part": part_folder,
                }
            )

    return items


def collect_unknown_images(unknown_path):
    items = []

    if not os.path.exists(unknown_path):
        return items

    for filename in sorted(os.listdir(unknown_path)):
        if not is_image_file(filename):
            continue

        image_path = os.path.join(unknown_path, filename)
        items.append(image_path)

    return items


def run_known_tests(recognizer):
    print("\nRunning known-part tests...")

    dataset_images = collect_dataset_images(DATASET_PATH)
    rows = []

    correct = 0
    total = 0

    for item in dataset_images:
        image_path = item["image_path"]
        expected_part = item["expected_part"]

        img = cv2.imread(image_path)

        if img is None:
            rows.append(
                {
                    "test_type": "known",
                    "image_path": image_path,
                    "expected": expected_part,
                    "predicted": "Image Load Error",
                    "flag": 0,
                    "passed": "No",
                    "reason": "Could not read image",
                    "best_score": 0,
                    "second_best_score": 0,
                    "raw_matches": 0,
                    "inlier_ratio": 0,
                    "blur_score": 0,
                    "time_seconds": 0,
                }
            )
            continue

        start = time.time()

        result = recognizer.recognize_image(
            test_img=img,
            threshold=RECOGNITION_THRESHOLD,
            separation_ratio=SEPARATION_RATIO,
            min_inlier_ratio=MIN_INLIER_RATIO,
            blur_threshold=BLUR_THRESHOLD,
            check_blur=True,
            use_ransac=True,
            skip_image_path=image_path,
        )

        elapsed = time.time() - start

        predicted = result["part"]
        passed = result["flag"] == 1 and predicted == expected_part

        total += 1

        if passed:
            correct += 1

        rows.append(
            {
                "test_type": "known_leave_one_out",
                "image_path": image_path,
                "expected": expected_part,
                "predicted": predicted,
                "flag": result["flag"],
                "passed": "Yes" if passed else "No",
                "reason": result["reason"],
                "best_score": result["best_score"],
                "second_best_score": result["second_best_score"],
                "raw_matches": result["raw_matches"],
                "inlier_ratio": f"{result['inlier_ratio']:.2f}",
                "blur_score": f"{result['blur_score']:.2f}",
                "time_seconds": f"{elapsed:.3f}",
            }
        )

    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"Known tests passed: {correct}/{total}")
    print(f"Known-part accuracy: {accuracy:.2f}%")

    return rows, correct, total


def run_unknown_tests(recognizer):
    print("\nRunning unknown-object tests...")

    unknown_images = collect_unknown_images(UNKNOWN_TESTS_PATH)
    rows = []

    correct_rejections = 0
    total = 0

    if not unknown_images:
        print("No unknown_tests folder/images found.")
        print("Create folder 'unknown_tests' and add unknown object images for rejection testing.")
        return rows, correct_rejections, total

    for image_path in unknown_images:
        img = cv2.imread(image_path)

        if img is None:
            rows.append(
                {
                    "test_type": "unknown",
                    "image_path": image_path,
                    "expected": "Unknown",
                    "predicted": "Image Load Error",
                    "flag": 0,
                    "passed": "No",
                    "reason": "Could not read image",
                    "best_score": 0,
                    "second_best_score": 0,
                    "raw_matches": 0,
                    "inlier_ratio": 0,
                    "blur_score": 0,
                    "time_seconds": 0,
                }
            )
            continue

        start = time.time()

        result = recognizer.recognize_image(
            test_img=img,
            threshold=RECOGNITION_THRESHOLD,
            separation_ratio=SEPARATION_RATIO,
            min_inlier_ratio=MIN_INLIER_RATIO,
            blur_threshold=BLUR_THRESHOLD,
            check_blur=True,
            use_ransac=True,
        )

        elapsed = time.time() - start

        predicted = result["part"]
        passed = result["flag"] == 0

        total += 1

        if passed:
            correct_rejections += 1

        rows.append(
            {
                "test_type": "unknown",
                "image_path": image_path,
                "expected": "Unknown",
                "predicted": predicted,
                "flag": result["flag"],
                "passed": "Yes" if passed else "No",
                "reason": result["reason"],
                "best_score": result["best_score"],
                "second_best_score": result["second_best_score"],
                "raw_matches": result["raw_matches"],
                "inlier_ratio": f"{result['inlier_ratio']:.2f}",
                "blur_score": f"{result['blur_score']:.2f}",
                "time_seconds": f"{elapsed:.3f}",
            }
        )

    rejection_rate = (correct_rejections / total) * 100 if total > 0 else 0

    print(f"Unknown tests rejected correctly: {correct_rejections}/{total}")
    print(f"Unknown rejection rate: {rejection_rate:.2f}%")

    return rows, correct_rejections, total


def save_report(rows):
    if not rows:
        print("\nNo report rows to save.")
        return

    fieldnames = [
        "test_type",
        "image_path",
        "expected",
        "predicted",
        "flag",
        "passed",
        "reason",
        "best_score",
        "second_best_score",
        "raw_matches",
        "inlier_ratio",
        "blur_score",
        "time_seconds",
    ]

    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTest report saved at:\n{REPORT_PATH}")


def main():
    if not os.path.exists(DATASET_PATH):
        print("Dataset folder not found:", DATASET_PATH)
        return

    print("Building dataset descriptor index...")

    recognizer = get_recognizer(
        DATASET_PATH,
        PARTS_CSV_PATH,
        force_rebuild=True,
    )

    print("Dataset parts:", recognizer.index.total_parts)
    print("Usable dataset images:", recognizer.index.total_images)
    print("Index build time:", f"{recognizer.index.build_time_seconds:.2f}s")

    if recognizer.index.load_errors:
        print("\nDataset warnings:")
        for error in recognizer.index.load_errors[:10]:
            print("-", error)

    all_rows = []

    known_rows, known_correct, known_total = run_known_tests(recognizer)
    unknown_rows, unknown_correct, unknown_total = run_unknown_tests(recognizer)

    all_rows.extend(known_rows)
    all_rows.extend(unknown_rows)

    save_report(all_rows)

    print("\nFinal Summary")
    print("-------------")

    if known_total > 0:
        print(f"Known accuracy: {(known_correct / known_total) * 100:.2f}%")

    if unknown_total > 0:
        print(f"Unknown rejection rate: {(unknown_correct / unknown_total) * 100:.2f}%")

    print("\nTesting complete.")


if __name__ == "__main__":
    main()