import cv2
import json
import pyautogui
import numpy as np
import time

# ===== CONFIGURATION =====
OUTPUT_FILE = "zones.json"
SCREEN_REGION = (0, 0, 1280, 720)  # Adjust if your game window is different

# --- Zone Names ---
ZONE_NAMES = [
    "gameplay_area",
    "note_lane_1",
    "note_lane_2",
    "note_lane_3",
    "note_lane_4",
    "hit_point_1",
    "hit_point_2",
    "hit_point_3",
    "hit_point_4",
    "accuracy_zone"
]

# --- Judgment “Points” (now rectangles) ---
JUDGMENT_NAMES = [
    "miss_zone",
    "point_50_zone",
    "point_100_zone",
    "point_200_zone",
    "point_300_zone",
    "point_300plus_zone"
]

zones = {}
drawing = False
start_point = None
current_name = None
image = None
clone = None


def click_event_rect(event, x, y, flags, param):
    """Handles rectangle selection for both zones and judgments."""
    global drawing, start_point, zones, image, current_name
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = image.copy()
        cv2.rectangle(temp, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Select Zones", temp)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        x1, y1 = min(start_point[0], end_point[0]), min(start_point[1], end_point[1])
        x2, y2 = max(start_point[0], end_point[0]), max(start_point[1], end_point[1])
        w, h = x2 - x1, y2 - y1
        zones[current_name] = {"x": x1, "y": y1, "w": w, "h": h}
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, current_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Select Zones", image)
        print(f"✅ Selected {current_name}: x={x1}, y={y1}, w={w}, h={h}")


def main():
    global image, clone, current_name

    print("⏳ You have 3 seconds to switch to your Osu!mania window...")
    time.sleep(3)

    print("📸 Capturing screenshot of region:", SCREEN_REGION)
    screenshot = pyautogui.screenshot(region=SCREEN_REGION)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    image = screenshot.copy()
    clone = image.copy()

    cv2.namedWindow("Select Zones", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Zones", 1280, 720)

    # Step 1: draw main gameplay zones
    for name in ZONE_NAMES:
        current_name = name
        print(f"\n👉 Draw rectangle for: {name}")
        print("   Click and drag, then release the mouse.")
        cv2.setMouseCallback("Select Zones", click_event_rect)
        cv2.imshow("Select Zones", image)
        cv2.waitKey(0)

    # Step 2: draw judgment areas (small rectangles)
    print("\n🎯 Now draw boxes for judgment areas (miss, 50, 100, etc.)")
    for name in JUDGMENT_NAMES:
        current_name = name
        print(f"\n👉 Draw rectangle for: {name}")
        cv2.setMouseCallback("Select Zones", click_event_rect)
        cv2.imshow("Select Zones", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Save everything
    output = {"zones": zones}
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\n💾 Saved all {len(zones)} zones to {OUTPUT_FILE}")
    print("\n📋 Summary:")
    for k, v in zones.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
