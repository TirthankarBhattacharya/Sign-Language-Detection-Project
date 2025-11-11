import os
import time
import csv
import cv2
import argparse
from utils.hand_landmarks import HandLandmarkExtractor

def main():
    parser = argparse.ArgumentParser(description="Collect hand landmark samples for a label.")
    parser.add_argument("--label", required=True, help="Class/label name, e.g., Hello")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to capture")
    parser.add_argument("--outdir", default=os.path.join("..", "data", "raw"), help="Output directory for CSV")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    outfile = os.path.join(args.outdir, f"{args.label}_{int(time.time())}.csv")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    extractor = HandLandmarkExtractor()

    print("Controls:")
    print("  - Position your hand in frame; hold the desired sign.")
    print("  - Press SPACE to record a sample; 'q' to quit.")
    print(f"Saving to: {outfile}")

    saved = 0
    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        # header
        headers = [f"f{i}" for i in range(64)] + ["label"]
        writer.writerow(headers)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = extractor.process(frame)
            vis = frame.copy()
            vis = extractor.draw_landmarks(vis, results)

            cv2.putText(vis, f"Label: {args.label} | Saved: {saved}/{args.samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(vis, "SPACE = capture, Q = quit",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Collect Data", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                feats, handed = extractor.landmarks_to_features(results)
                if feats is not None:
                    row = list(feats) + [args.label]
                    writer.writerow(row)
                    saved += 1
                    if saved >= args.samples:
                        print("Reached target samples. Exiting.")
                        break
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} samples to {outfile}")

if __name__ == "__main__":
    main()
