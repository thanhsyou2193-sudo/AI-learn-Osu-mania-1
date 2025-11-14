# screen_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import pyautogui
import pytesseract
import json
import time
import keyboard
import re
import os


class OsuManiaScreenEnv(gym.Env):
    def __init__(self, zones_path="zones.json", frame_size=(84, 84), show_debug=False,
                 start_on_first_note=True, start_delay_ms=0,
                 gate_actions_by_detection=False, max_simultaneous_keys=4,
                 # note detection tuning (defaults to older/simple detector)
                 detect_band_px=24, detect_abs_thresh=70, detect_ratio_thresh=0.02, use_otsu=False,
                 pre_hit_offset_px=1,
                 # song gate using "point zone" black screen (enabled by default)
                 use_point_zone_gate=True, point_zone_key="point_300_zone", point_zone_black_thresh=25,
                 # debug window placement
                 debug_window_position="bottom_left",
                 # debug judgement detection
                 save_judgment_debug=False,
                 # long-note behavior
                 hold_requires_tap=True,
                 # hit window gating (disable for testing if agent never presses)
                 require_hit_window=True,
                 # judgement -> reward scaling (enabled by default)
                 judgement_reward_scale=1.0):
        super().__init__()
        self.keys = ['d', 'f', 'j', 'k']
        self.action_space = spaces.MultiBinary(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=frame_size, dtype=np.uint8)

        with open(zones_path, "r") as f:
            self.zones = json.load(f)["zones"]

        # Absolute gameplay area (desktop coords)
        self.gameplay = self.zones["gameplay_area"]
        self.screen_region = (
            self.gameplay["x"],
            self.gameplay["y"],
            self.gameplay["w"],
            self.gameplay["h"]
        )

        self.frame_size = frame_size
        self.intensity = np.zeros(4)
        self.hit_flashes = np.zeros(4)
        self.combo = 0
        self.step_count = 0
        self.cooldowns = np.zeros(4)
        self.no_activity_counter = 0
        self.show_debug = show_debug
        self._dbg_last_ts = None
        self._dbg_fps = 0.0
        self._dbg_hits = 0
        self._dbg_misses = 0
        self._dbg_win_name = "OsuAI Debug"
        self._dbg_window_initialized = False
        self.debug_window_position = str(debug_window_position)
        
        # Key state for true holds (for long notes)
        self._key_down = np.zeros(4, dtype=bool)
        self._ln_hold = np.zeros(4, dtype=bool)
        self._ln_release_grace_ctr = np.zeros(4, dtype=int)
        # detection flash + counters
        self._lane_detect_flash = np.zeros(4, dtype=int)
        self._prev_active_near_hit = np.zeros(4, dtype=bool)
        self.notes_seen = 0

        # Start timing on first note detection near hit line
        self.start_on_first_note = start_on_first_note
        self.start_delay_ms = start_delay_ms
        self._armed = True  # waiting for first note
        self._start_deadline = None

        # Action gating to avoid pressing many lanes when only 1 note exists
        self.gate_actions_by_detection = gate_actions_by_detection
        self.max_simultaneous_keys = max(1, int(max_simultaneous_keys))

        # Detection params
        self.detect_band_px = int(detect_band_px)
        self.detect_abs_thresh = float(detect_abs_thresh)
        self.detect_ratio_thresh = float(detect_ratio_thresh)
        self.use_otsu = bool(use_otsu)
        self.pre_hit_offset_px = int(pre_hit_offset_px)

        # Song gating by point zone (black -> playing)
        self.use_point_zone_gate = bool(use_point_zone_gate)
        self.point_zone_key = str(point_zone_key)
        self.point_zone_black_thresh = float(point_zone_black_thresh)
        self._song_active = False
        
        # Long note sustain detection (below hit line)
        self.sustain_band_px = 32
        self.sustain_abs_thresh = 60.0
        self.hold_release_grace = 2
        self.ln_start_confirm_frames = 4  # require body below hit for N frames to start hold
        self.ln_end_confirm_frames = 4    # require missing body for N frames to end hold
        self.ln_max_hold_frames = 120     # safety cap in frames (~2s at 60fps)
        self.lane_shrink_px = 5           # shrink lane width to reduce crosstalk
        self.hold_requires_tap = bool(hold_requires_tap)
        self._hold_started_from_tap = np.zeros(4, dtype=bool)
        self.require_hit_window = bool(require_hit_window)

        # Judgement tracking / reward scaling
        self.judgement_reward_scale = float(judgement_reward_scale)
        self.pred_score = 0
        self.judge_counts = {"miss": 0, "50": 0, "100": 0, "200": 0, "300": 0, "300g": 0}
        self.last_judgement = None
        self._last_judge_step = 0
        
        # Hit window: only allow presses when notes are at hit line
        self.hit_window_band = 12  # px band around hit zone for timing (increased for more lenient timing)

        os.makedirs("ocr_debug", exist_ok=True)
        self.save_judgment_debug = bool(save_judgment_debug)
        if self.save_judgment_debug:
            os.makedirs("judgment_debug", exist_ok=True)
        self._judge_debug_ctr = 0

        # Throttle OCR to avoid FPS impact
        self._ocr_stride = 30
        self._last_acc = None

        try:
            pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            self.tesseract_available = True
        except Exception:
            print("⚠️ Tesseract not found, disabling OCR.")
            self.tesseract_available = False

        # Fast screen capture backends (Windows): prefer dxcam, then mss, then pyautogui
        self._cap_backend = "pyautogui"
        self._dxcam = None
        self._mss = None

        # Load judgement templates
        self._templates = self._load_judgement_templates()
        try:
            import dxcam  # type: ignore
            self._dxcam = dxcam.create(output_idx=0)
            self._cap_backend = "dxcam"
            print("📸 Using dxcam for capture")
        except Exception:
            try:
                import mss  # type: ignore
                self._mss = mss.mss()
                self._cap_backend = "mss"
                print("📸 Using mss for capture")
            except Exception:
                print("📸 Falling back to pyautogui capture (slow)")

    # ---------------- ENV CORE ----------------
    def reset(self, **kwargs):
        self.intensity = np.zeros(4)
        self.hit_flashes = np.zeros(4)
        self.combo = 0
        self.step_count = 0
        self.cooldowns = np.zeros(4)
        self.no_activity_counter = 0
        self._dbg_last_ts = time.perf_counter()
        self._dbg_fps = 0.0
        self._dbg_hits = 0
        self._dbg_misses = 0
        self._armed = self.start_on_first_note
        self._start_deadline = None
        self._last_acc = None
        self._song_active = False
        self.pred_score = 0
        self.judge_counts = {"miss": 0, "50": 0, "100": 0, "200": 0, "300": 0, "300g": 0}
        self.last_judgement = None
        self._last_judge_step = 0
        self._lane_detect_flash = np.zeros(4, dtype=int)
        self._prev_active_near_hit = np.zeros(4, dtype=bool)
        self.notes_seen = 0
        self._key_down[:] = False
        self._ln_hold[:] = False
        self._ln_release_grace_ctr[:] = 0
        self._below_active_ctr = np.zeros(4, dtype=int)
        self._hold_frames = np.zeros(4, dtype=int)
        self._hold_started_from_tap[:] = False

        screenshot = self._capture_screen()
        if screenshot.size == 0:
            print("⚠️ Failed to capture screenshot in reset.")
            return np.zeros(self.frame_size, dtype=np.uint8), {}

        obs = self._preprocess_observation(screenshot)
        if self.show_debug:
            self._show_debug(screenshot, np.zeros(4, dtype=bool), np.zeros(4, dtype=int), [0,0,0,0], {})
            self._dbg_window_initialized = True
        return obs, {}

    def step(self, action):
        self.step_count += 1
        screenshot = self._capture_screen()

        if screenshot.size == 0:
            print("⚠️ Empty screenshot detected.")
            obs = np.zeros(self.frame_size, dtype=np.uint8)
            return obs, -1.0, False, True, {}

        # 🎯 Detect results screen
        if self.is_result_screen(screenshot):
            print("🎯 Results screen detected — ending episode.")
            obs = np.zeros(self.frame_size, dtype=np.uint8)
            return obs, 0.0, True, False, {}

        # --- Detect end of song (no note activity) ---
        self.intensity = self._calculate_intensities(screenshot)
        self.hit_flashes = self._get_hit_flashes(screenshot)

        # Point-zone based gate for song start/stop
        if self.use_point_zone_gate:
            p_brightness = self._point_zone_brightness(screenshot)
            is_black = p_brightness < self.point_zone_black_thresh
            if is_black and not self._song_active:
                self._song_active = True  # song started
            if self._song_active and not is_black:
                print("🎵 Point zone left black — song ended.")
                self._print_episode_summary()
                obs = np.zeros(self.frame_size, dtype=np.uint8)
                return obs, 0.0, True, False, {}

        # Fallback end detection (silence)
        if np.max(self.intensity) < 20 and np.max(self.hit_flashes) < 20:
            self.no_activity_counter += 1
        else:
            self.no_activity_counter = 0

        if self.no_activity_counter > 90:
            print("🎵 No more notes detected — song ended (fallback).")
            self._print_episode_summary()
            obs = np.zeros(self.frame_size, dtype=np.uint8)
            return obs, 0.0, True, False, {}

        # Detect notes in different zones
        active_near_hit = self._detect_pre_hit_notes(screenshot)  # above hit, for early warning
        in_hit_window = self._detect_in_hit_window(screenshot)   # at hit line, for precise timing
        below_hit_active = self._detect_below_hit_body(screenshot)  # below hit, for LN sustain
        # rising-edge detection for visualization and counting
        rising = np.logical_and(active_near_hit, np.logical_not(self._prev_active_near_hit))
        if np.any(rising):
            self._lane_detect_flash[rising] = 8  # flash for ~8 frames
            self.notes_seen += int(np.sum(rising))
        # update below-hit activity counters
        for i in range(4):
            if below_hit_active[i]:
                self._below_active_ctr[i] += 1
            else:
                self._below_active_ctr[i] = 0

        self._prev_active_near_hit = active_near_hit.copy()

        now = time.perf_counter()
        if self._armed and np.any(active_near_hit):
            self._start_deadline = now + (self.start_delay_ms / 1000.0)
            self._armed = False
        allow_press = (not self.start_on_first_note) or (self._start_deadline is not None and now >= self._start_deadline)
        if self.use_point_zone_gate:
            allow_press = allow_press and self._song_active

        # Gate action by detection (optional) and limit max simultaneous keys
        orig_mask = np.array(action, dtype=bool)
        gated_mask = orig_mask.copy()
        if self.gate_actions_by_detection:
            # Only allow keys where a note is near the hit line
            gated_mask = gated_mask & active_near_hit
            # If still too many, keep top by lane intensity
            if gated_mask.sum() > self.max_simultaneous_keys:
                # Rank lanes by current intensity descending
                idxs = np.argsort(-self.intensity)  # high to low
                keep = []
                for i in idxs:
                    if gated_mask[i]:
                        keep.append(i)
                        if len(keep) >= self.max_simultaneous_keys:
                            break
                new_mask = np.zeros_like(gated_mask)
                new_mask[keep] = True
                gated_mask = new_mask
        gated_action = gated_mask.astype(int)

        # Build tap mask (from policy) and hold mask (from LN detection)
        # Gate taps: only allow if note is in hit window for that lane (if enabled)
        if self.require_hit_window:
            tap_mask = gated_action.astype(bool) & in_hit_window
        else:
            tap_mask = gated_action.astype(bool)
        hold_mask = self._ln_hold.astype(bool)

        # Start/stop holds using tap knowledge to avoid false holds on singles
        for i in range(4):
            # Only allow starting hold if we tapped this lane in window AND body detected
            if self._ln_hold[i] is False and tap_mask[i] is True and below_hit_active[i] and self._below_active_ctr[i] >= self.ln_start_confirm_frames:
                self._ln_hold[i] = True
                self._hold_started_from_tap[i] = True
                self._hold_frames[i] = 0
                self._ln_release_grace_ctr[i] = 0
            
            # While holding: update timers and release when body ends
            if self._ln_hold[i]:
                self._hold_frames[i] += 1
                if below_hit_active[i]:
                    self._ln_release_grace_ctr[i] = 0
                else:
                    self._ln_release_grace_ctr[i] += 1
                # Release if body missing for N frames OR hit max duration
                if self._ln_release_grace_ctr[i] >= self.ln_end_confirm_frames or self._hold_frames[i] >= self.ln_max_hold_frames:
                    self._ln_hold[i] = False
                    self._hold_started_from_tap[i] = False
                    self._hold_frames[i] = 0
                    self._ln_release_grace_ctr[i] = 0

        # Press keys only if allowed
        if allow_press:
            self._press_keys(tap_mask, hold_mask)
        else:
            # Release any holds if we are not allowed to press
            if np.any(self._key_down):
                self._press_keys(np.zeros(4, dtype=bool), np.zeros(4, dtype=bool))
            time.sleep(0.002)

        # Base reward from lane intensity (minor)
        base_reward = self._calculate_reward(gated_action) * 0.1

        # Detect judgement and use for primary reward signal
        judge_name, judge_points = self._detect_judgement_if_any(screenshot)
        judge_reward = 0.0
        if judge_name is not None:
            self.last_judgement = judge_name
            self.judge_counts[judge_name] += 1
            self.pred_score += int(judge_points)
            self._last_judge_step = self.step_count
            # Map judgment to reward (300g=+2, 300=+1.5, 200=+1, 100=+0.3, 50=-0.2, miss=-1)
            judge_map = {"300g": 2.0, "300": 1.5, "200": 1.0, "100": 0.3, "50": -0.2, "miss": -1.0}
            judge_reward = judge_map.get(judge_name, 0.0) * self.judgement_reward_scale
        
        reward = base_reward + judge_reward

        obs = self._preprocess_observation(screenshot)

        # Debug hit/miss accounting for overlay
        active_lanes = self.intensity > 50
        vis_pressed = np.logical_or(tap_mask, hold_mask)
        hits = int(np.sum((vis_pressed == True) & (active_lanes == True)))
        misses = int(np.sum((vis_pressed == False) & (active_lanes == True)))
        self._dbg_hits += hits
        self._dbg_misses += misses
        hold_dbg = hold_mask.astype(int).tolist()

        # decay flash timers
        self._lane_detect_flash = np.maximum(0, self._lane_detect_flash - 1)

        # Throttled OCR (expensive)
        acc = None
        if self.tesseract_available and (self.step_count % self._ocr_stride == 0):
            acc = self._get_accuracy(screenshot)
            self._last_acc = acc
        else:
            acc = self._last_acc

        info = {
            "intensity": self.intensity.tolist(),
            "combo": self.combo,
            "accuracy": acc,
            "hit_flash": self.hit_flashes.tolist(),
            "fps": round(self._dbg_fps, 1),
            "armed": self._armed,
            "start_deadline": self._start_deadline,
            "hits": self._dbg_hits,
            "misses": self._dbg_misses,
            "orig_action": orig_mask.astype(int).tolist(),
            "gated_action": gated_action.tolist(),
            "hold_mask": hold_mask.astype(int).tolist(),
            "score": int(self.pred_score),
            "last_judgement": self.last_judgement,
            "judge_counts": self.judge_counts,
            "song_active": self._song_active,
            "notes_seen": int(self.notes_seen),
            "active_near_hit": active_near_hit.astype(int).tolist(),
            "in_hit_window": in_hit_window.astype(int).tolist(),
            "hold_dbg": hold_dbg,
        }

        if self.show_debug:
            self._show_debug(screenshot, active_near_hit, vis_pressed, hold_dbg, info)

        return obs, reward, False, False, info

    # ---------------- DETECTION ----------------
    def is_result_screen(self, screen):
        h, w = screen.shape[:2]
        region = screen[int(h * 0.25):int(h * 0.6), int(w * 0.25):int(w * 0.75)]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness > 160:
            text = pytesseract.image_to_string(region, config="--psm 6").lower()
            if "accuracy" in text or "reessayer" in text or "rejouer" in text:
                print(f"OCR detected result screen: {text.strip()[:80]}")
                return True
        return False

    # ---------------- HELPERS ----------------
    def _rel(self, zone):
        """Convert absolute desktop coords to local gameplay image coords."""
        return {
            "x": int(zone["x"] - self.gameplay["x"]),
            "y": int(zone["y"] - self.gameplay["y"]),
            "w": int(zone["w"]),
            "h": int(zone["h"]),
        }

    def _press_keys(self, tap_mask, hold_mask):
        """Tap keys in tap_mask; maintain holds in hold_mask using true key down state."""
        taps_to_release = []
        # maintain holds and releases
        for i in range(4):
            if hold_mask[i]:
                if not self._key_down[i]:
                    keyboard.press(self.keys[i])
                    self._key_down[i] = True
            else:
                if self._key_down[i]:
                    keyboard.release(self.keys[i])
                    self._key_down[i] = False
        # do taps (avoid tapping keys that are held)
        for i in range(4):
            if tap_mask[i] and (not hold_mask[i]) and self.cooldowns[i] <= 0:
                keyboard.press(self.keys[i])
                taps_to_release.append(i)
                self.cooldowns[i] = 4
        if taps_to_release:
            time.sleep(0.008)
            for i in taps_to_release:
                keyboard.release(self.keys[i])
        # cool down
        self.cooldowns = np.maximum(0, self.cooldowns - 1)

    def _capture_screen(self):
        x, y, w, h = self.screen_region
        try:
            if self._cap_backend == "dxcam" and self._dxcam is not None:
                frame = self._dxcam.grab(region=(x, y, x + w, y + h))
                if frame is None:
                    raise RuntimeError("dxcam returned None")
                # dxcam returns BGRA
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif self._cap_backend == "mss" and self._mss is not None:
                monitor = {"left": x, "top": y, "width": w, "height": h}
                shot = self._mss.grab(monitor)
                img = np.array(shot)  # BGRA
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                shot = pyautogui.screenshot(region=(x, y, w, h))
                return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Screenshot failed: {e}")
            return np.zeros((h, w, 3), dtype=np.uint8)

    def _preprocess_observation(self, screen):
        # screen is already the gameplay area; just grayscale and resize
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)

    def _calculate_intensities(self, screen):
        vals = []
        for i in range(1, 5):
            lane_abs = self.zones[f"note_lane_{i}"]
            lane = self._rel(lane_abs)
            part = screen[lane["y"]:lane["y"] + lane["h"], lane["x"]:lane["x"] + lane["w"]]
            gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
            vals.append(np.mean(gray))
        arr = np.clip(np.array(vals), 0, 255)
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-5) * 255

    def _get_hit_flashes(self, screen):
        flashes = []
        for i in range(1, 5):
            zone_abs = self.zones[f"hit_point_{i}"]
            zone = self._rel(zone_abs)
            crop = screen[zone["y"]:zone["y"] + zone["h"], zone["x"]:zone["x"] + zone["w"]]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # use top-k brightest pixels to sense flash
            flat = gray.reshape(-1)
            k = min(50, flat.size)
            top_pixels = np.partition(flat, -k)[-k:]
            flashes.append(np.mean(top_pixels))
        return np.array(flashes)

    def _calculate_reward(self, action):
        reward = 0.0
        active_lanes = self.intensity > 50

        for i in range(4):
            if action[i] == 1 and active_lanes[i]:
                reward += 1.0
                self.combo += 1
            elif action[i] == 1 and not active_lanes[i]:
                reward -= 0.4
                self.combo = 0

        if np.sum(action) >= 3:
            reward -= 0.5
            self.combo = 0

        if np.any(active_lanes) and np.sum(action) == 0:
            reward -= 0.3
            self.combo = 0

        reward += min(self.combo / 100.0, 0.5)
        return float(np.clip(reward, -2.0, 3.0))

    def _rel_shrink(self, zone, shrink_x):
        z = self._rel(zone)
        sx = int(max(0, shrink_x))
        x = z["x"] + sx
        w = max(1, z["w"] - 2 * sx)
        return {"x": x, "y": z["y"], "w": w, "h": z["h"]}

    def _detect_in_hit_window(self, screen):
        """Detect notes inside the hit window at the hit line for timing.
        Window now spans from above hit zone to below to catch notes crossing.
        """
        active = []
        for i in range(1, 5):
            lane_abs = self.zones[f"note_lane_{i}"]
            hit_abs = self.zones[f"hit_point_{i}"]
            lane = self._rel_shrink(lane_abs, self.lane_shrink_px)
            hit = self._rel(hit_abs)
            # window: from N px above hit zone to N px below
            y1 = max(hit["y"] - self.hit_window_band, 0)
            y2 = min(hit["y"] + hit["h"] + self.hit_window_band, self.gameplay["h"] - 1)
            band = screen[y1:y2, lane["x"]:lane["x"] + lane["w"]]
            if band.size == 0:
                active.append(False)
                continue
            gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            is_active = float(np.mean(gray)) >= self.detect_abs_thresh
            active.append(is_active)
        return np.array(active, dtype=bool)

    def _detect_below_hit_body(self, screen):
        """Detect note body below the hit line (sustain for long notes)."""
        active = []
        for i in range(1, 5):
            lane_abs = self.zones[f"note_lane_{i}"]
            hit_abs = self.zones[f"hit_point_{i}"]
            lane = self._rel_shrink(lane_abs, self.lane_shrink_px)
            hit = self._rel(hit_abs)
            y1 = hit["y"] + hit["h"]
            y2 = min(y1 + self.sustain_band_px, self.gameplay["h"] - 1)
            band = screen[y1:y2, lane["x"]:lane["x"] + lane["w"]]
            if band.size == 0:
                active.append(False)
                continue
            gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            is_active = float(np.mean(gray)) >= self.sustain_abs_thresh
            active.append(is_active)
        return np.array(active, dtype=bool)

    def _detect_pre_hit_notes(self, screen):
        """Detect notes in a band above each hit line.
        - If use_otsu=True: Otsu + white ratio threshold
        - Else: simple mean brightness threshold (older behavior)
        """
        active = []
        band_px = self.detect_band_px
        for i in range(1, 5):
            lane_abs = self.zones[f"note_lane_{i}"]
            hit_abs = self.zones[f"hit_point_{i}"]
            lane = self._rel_shrink(lane_abs, self.lane_shrink_px)
            hit = self._rel(hit_abs)
            y2 = max(hit["y"] - self.pre_hit_offset_px, 0)
            y1 = max(y2 - band_px, 0)
            band = screen[y1:y2, lane["x"]:lane["x"] + lane["w"]]
            if band.size == 0:
                active.append(False)
                continue
            gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            if self.use_otsu:
                _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                white_ratio = float(np.count_nonzero(binimg)) / float(binimg.size)
                is_active = white_ratio >= self.detect_ratio_thresh
            else:
                is_active = float(np.mean(gray)) >= self.detect_abs_thresh
            active.append(is_active)
        return np.array(active, dtype=bool)

    def _point_zone_brightness(self, screen):
        zone_abs = self.zones.get(self.point_zone_key, None)
        if not zone_abs:
            return 255.0
        zone = self._rel(zone_abs)
        crop = screen[zone["y"]:zone["y"] + zone["h"], zone["x"]:zone["x"] + zone["w"]]
        if crop.size == 0:
            return 255.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _get_accuracy(self, screen):
        try:
            zone = self._rel(self.zones["accuracy_zone"])
            region = screen[zone["y"]:zone["y"] + zone["h"], zone["x"]:zone["x"] + zone["w"]]
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(
                gray, config="--psm 7 -c tessedit_char_whitelist=0123456789.%"
            ).strip()
            text = text.replace("O", "0").replace("o", "0")
            # Match format: XX.XX% or XXX.XX%
            match = re.search(r"(\d{1,3}\.\d{1,2})", text)
            if match:
                val = float(match.group(1))
                # clamp to 0-100
                return min(100.0, max(0.0, val))
        except Exception:
            return None
        return None

    def _ensure_debug_window(self, img_w, img_h):
        try:
            screen_w, screen_h = pyautogui.size()
        except Exception:
            screen_w, screen_h = (1920, 1080)
        margin_x, margin_y = 20, 20
        if self.debug_window_position == "top_right":
            x = max(0, screen_w - img_w - margin_x)
            y = max(0, margin_y)
        elif self.debug_window_position == "bottom_right":
            x = max(0, screen_w - img_w - margin_x)
            y = max(0, screen_h - img_h - margin_y)
        elif self.debug_window_position == "bottom_left":
            x = max(0, margin_x)
            y = max(0, screen_h - img_h - margin_y)
        else:  # fallback bottom-left
            x = max(0, margin_x)
            y = max(0, screen_h - img_h - margin_y)
        cv2.namedWindow(self._dbg_win_name, cv2.WINDOW_NORMAL)
        try:
            cv2.moveWindow(self._dbg_win_name, x, y)
        except Exception:
            pass

    def _load_one_template(self, filename):
        # Try absolute path then relative 'templates' next to this file
        if os.path.isabs(filename) and os.path.exists(filename):
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            return img
        base_dir = os.path.join(os.path.dirname(__file__), 'templates')
        alt = os.path.join(base_dir, os.path.basename(filename))
        if os.path.exists(alt):
            return cv2.imread(alt, cv2.IMREAD_GRAYSCALE)
        return None

    def _load_judgement_templates(self):
        # Mapping of zones to (name, template_path, points)
        mapping = {
            "miss_zone": ("miss", r"C:\\Users\\Thanh\\Desktop\\OsuAI\\templates\\mania-hit0-0.png", 0),
            "point_50_zone": ("50", r"C:\\Users\\Thanh\\Desktop\\OsuAI\\templates\\mania-hit50-0.png", 50),
            "point_100_zone": ("100", r"C:\\Users\\Thanh\\Desktop\\OsuAI\\templates\\mania-hit100-0.png", 100),
            "point_200_zone": ("200", r"C:\\Users\\Thanh\\Desktop\\OsuAI\\templates\\mania-hit200-0.png", 200),
            "point_300_zone": ("300", r"C:\\Users\\Thanh\\Desktop\\OsuAI\\templates\\mania-hit300-0.png", 300),
            "point_300plus_zone": ("300g", r"C:\\Users\\Thanh\\Desktop\\OsuAI\\templates\\mania-hit300g.png", 300),
        }
        out = {}
        for zone_key, (name, path, pts) in mapping.items():
            tmpl = self._load_one_template(path)
            if tmpl is None:
                print(f"⚠️ Template not found for {zone_key}: {path}")
            out[zone_key] = {"name": name, "points": pts, "img": tmpl}
        return out

    def _detect_judgement_if_any(self, screen, tm_thresh=0.55):
        # Trigger detection when hit flashes are bright or periodically
        if np.max(self.hit_flashes) < 160 and (self.step_count % 3 != 0):
            return None, 0
        best_name, best_pts, best_val = None, 0, 0.0
        debug_scores = {}
        for zone_key, meta in self._templates.items():
            tmpl = meta.get("img", None)
            if tmpl is None:
                continue
            zone_abs = self.zones.get(zone_key, None)
            if not zone_abs:
                continue
            zone = self._rel(zone_abs)
            crop = screen[zone["y"]:zone["y"] + zone["h"], zone["x"]:zone["x"] + zone["w"]]
            if crop.size == 0 or crop.shape[0] < tmpl.shape[0] or crop.shape[1] < tmpl.shape[1]:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # resize crop to template size if slightly different
            if gray.shape != tmpl.shape:
                gray = cv2.resize(gray, (tmpl.shape[1], tmpl.shape[0]), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            debug_scores[meta["name"]] = max_val
            
            # Save debug image if enabled
            if self.save_judgment_debug and np.max(self.hit_flashes) > 180:
                self._judge_debug_ctr += 1
                dbg_path = f"judgment_debug/step{self.step_count}_{zone_key}_{meta['name']}_score{max_val:.3f}.png"
                cv2.imwrite(dbg_path, gray)
            
            if max_val > tm_thresh and max_val > best_val:
                best_val = max_val
                best_name = meta["name"]
                best_pts = meta["points"]
        
        # Log scores if debug enabled and bright flash
        if self.save_judgment_debug and np.max(self.hit_flashes) > 180:
            print(f"  Judge scores: {debug_scores}")
        
        if best_name is not None:
            return best_name, best_pts
        return None, 0

    def _draw_osk(self, img, pressed_mask):
        # Draw a small on-screen keyboard for D F J K at bottom-right of the debug image
        h, w = img.shape[:2]
        key_w, key_h, pad = 44, 28, 8
        total_w = key_w * 4 + pad * 3
        x0 = w - total_w - 16
        y0 = h - key_h - 16
        labels = [k.upper() for k in self.keys]
        for i, lab in enumerate(labels):
            x = x0 + i * (key_w + pad)
            y = y0
            pressed = bool(pressed_mask[i])
            color_fill = (0, 200, 0) if pressed else (40, 40, 40)
            color_border = (0, 255, 0) if pressed else (180, 180, 180)
            cv2.rectangle(img, (x, y), (x + key_w, y + key_h), color_fill, -1)
            cv2.rectangle(img, (x, y), (x + key_w, y + key_h), color_border, 2)
            cv2.putText(img, lab, (x + 12, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    def _show_debug(self, screen, active_near_hit, pressed_mask, hold_dbg, info=None):
        # compute FPS
        now = time.perf_counter()
        if self._dbg_last_ts is not None:
            dt = now - self._dbg_last_ts
            if dt > 0:
                self._dbg_fps = (self._dbg_fps * 0.8) + (0.2 * (1.0 / dt))
        self._dbg_last_ts = now

        img = screen.copy()

        # Draw lanes, hit points, and detection bands
        in_window = info.get("in_hit_window", [0,0,0,0])
        for i in range(1, 5):
            lane = self._rel(self.zones[f"note_lane_{i}"])
            hit = self._rel(self.zones[f"hit_point_{i}"])
            color_lane = (0, 255, 0)
            color_hit = (0, 200, 255)
            cv2.rectangle(img, (lane["x"], lane["y"]), (lane["x"]+lane["w"], lane["y"]+lane["h"]), color_lane, 1)
            cv2.rectangle(img, (hit["x"], hit["y"]), (hit["x"]+hit["w"], hit["y"]+hit["h"]), color_hit, 1)
            
            # pre-hit band (early warning)
            y2 = max(hit["y"] - self.pre_hit_offset_px, 0)
            y1 = max(y2 - self.detect_band_px, 0)
            c = (0, 0, 255) if active_near_hit[i-1] else (255, 0, 0)
            cv2.rectangle(img, (lane["x"], y1), (lane["x"]+lane["w"], y2), c, 2)
            
            # hit window band (where taps are allowed) - show in bright yellow if note detected
            hw_y1 = max(hit["y"] - self.hit_window_band, 0)
            hw_y2 = min(hit["y"] + hit["h"] + self.hit_window_band, img.shape[0] - 1)
            hw_color = (0, 255, 255) if in_window[i-1] else (100, 100, 0)
            cv2.rectangle(img, (lane["x"], hw_y1), (lane["x"]+lane["w"], hw_y2), hw_color, 2)

        # Flash highlight on hit line when detection triggers
        overlay = img.copy()
        for i in range(1, 5):
            if self._lane_detect_flash[i-1] > 0:
                hit = self._rel(self.zones[f"hit_point_{i}"])
                # draw bright line/box across hit area
                y1 = hit["y"]
                y2 = hit["y"] + hit["h"]
                x1 = hit["x"]
                x2 = hit["x"] + hit["w"]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        # blend to create glow
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

        # Overlay text
        status = "ACTIVE" if (not self._armed and (self._start_deadline is None or time.perf_counter() >= (self._start_deadline or 0))) else ("WAITING" if self._armed else "DELAY")
        gate_state = "PLAY" if self._song_active else "STOP"
        cv2.putText(img, f"FPS {self._dbg_fps:4.1f} | {status} | {gate_state}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img, f"Combo {self.combo} | Hits {self._dbg_hits} Miss {self._dbg_misses}", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img, f"Notes {self.notes_seen}", (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
        cv2.putText(img, f"Action {pressed_mask.astype(int).tolist()}", (8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        # score/judgement overlay
        if self.last_judgement is not None:
            cv2.putText(img, f"Last: {self.last_judgement}", (8, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        # Larger, prominent score display
        cv2.putText(img, f"AI SCORE: {self.pred_score:,}", (8, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 3)
        # Judgment counts
        jc_text = f"300g:{self.judge_counts.get('300g',0)} 300:{self.judge_counts.get('300',0)} 200:{self.judge_counts.get('200',0)} 100:{self.judge_counts.get('100',0)} 50:{self.judge_counts.get('50',0)} X:{self.judge_counts.get('miss',0)}"
        cv2.putText(img, jc_text, (8, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        # OSK
        self._draw_osk(img, pressed_mask)
        # Show hold lanes for clarity
        cv2.putText(img, f"Hold {hold_dbg}", (8, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # ensure window placement in bottom-right
        if not self._dbg_window_initialized:
            self._ensure_debug_window(img.shape[1], img.shape[0])
            self._dbg_window_initialized = True
        cv2.imshow(self._dbg_win_name, img)
        cv2.waitKey(1)

    def _print_episode_summary(self):
        print("\n" + "="*60)
        print("🏁 EPISODE SUMMARY")
        print("="*60)
        print(f"  AI Predicted Score: {self.pred_score:,}")
        print(f"  Total Notes Seen: {self.notes_seen}")
        print(f"  Combo: {self.combo}")
        print(f"\n  Judgements:")
        print(f"    300g: {self.judge_counts.get('300g', 0):4d}")
        print(f"    300:  {self.judge_counts.get('300', 0):4d}")
        print(f"    200:  {self.judge_counts.get('200', 0):4d}")
        print(f"    100:  {self.judge_counts.get('100', 0):4d}")
        print(f"    50:   {self.judge_counts.get('50', 0):4d}")
        print(f"    MISS: {self.judge_counts.get('miss', 0):4d}")
        total_judged = sum(self.judge_counts.values())
        print(f"  Total Judged: {total_judged}")
        if total_judged > 0:
            acc_300g = self.judge_counts.get('300g', 0) * 100.0 / total_judged
            acc_300 = self.judge_counts.get('300', 0) * 100.0 / total_judged
            print(f"  300g Rate: {acc_300g:.1f}%")
            print(f"  300 Rate: {acc_300:.1f}%")
        print("="*60 + "\n")

    def close(self):
        cv2.destroyAllWindows()
        self._dbg_window_initialized = False
