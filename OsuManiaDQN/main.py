# main.py
import torch
import numpy as np
import keyboard
import os
from screen_env import OsuManiaScreenEnv
from FrameStack import FrameStack
from dqn_agent import DQNAgent

MODEL_PATH = "osu_dqn_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment configuration (with brightness-based detection)
base_env = OsuManiaScreenEnv(
    show_debug=True,
    start_on_first_note=True,
    start_delay_ms=0,
    save_judgment_debug=False,

    detect_abs_thresh=50,
    pre_hit_offset_px=5,

    require_hit_window=True,
    use_point_zone_gate=False,

    judgement_reward_scale=2.0,

    # Brightness threshold for black detection
    point_zone_key="point_300_zone",
    point_zone_black_thresh=25,
)

env = FrameStack(base_env, num_stack=4)
state, info = env.reset()
state_tensor = torch.tensor(np.array(state), dtype=torch.float32).to(DEVICE)

agent = DQNAgent(input_shape=(4, 84, 84), action_dim=4, device=DEVICE)

# --- Load model if available ---
if os.path.exists(MODEL_PATH):
    agent.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    agent.target_model.load_state_dict(agent.model.state_dict())
    agent.epsilon = 0.7
    print(f"✅ Loaded model from {MODEL_PATH} (epsilon={agent.epsilon})")
else:
    agent.epsilon = 0.9
    print(f"🆕 Starting new model (epsilon={agent.epsilon})")

print("🎮 Starting Osu!mania AI (Press ESC to quit)")

# --- Variables for song state tracking ---
song_active_prev = False   # Previous frame brightness state (non-black?)
ai_enabled = False         # Whether AI is currently allowed to act

# --- Main Loop ---
for step in range(10000):
    if keyboard.is_pressed("esc"):
        print("🛑 ESC pressed — stopping manually.")
        break

    # Default action: idle
    action = 0
    if ai_enabled:
        action = agent.act(state_tensor)

    next_state, reward, terminated, truncated, info = env.step(action)

    # Get the current screen brightness detection flag
    song_active = info.get("song_active", False)
    # In this setup, song_active=True → screen is NON-BLACK
    # song_active=False → screen is BLACK

    # ✅ Song start: non-black → black
    if not song_active and song_active_prev:
        print("🎵 Detected transition NON-BLACK → BLACK — song START detected! AI ACTIVE 🎶")
        ai_enabled = True

    # ✅ Song end: black → non-black
    elif song_active and not song_active_prev:
        print("🖤 Detected transition BLACK → NON-BLACK — song END detected! Saving and exiting...")
        torch.save(agent.model.state_dict(), MODEL_PATH)
        break

    # Update state for next iteration
    song_active_prev = song_active

    # Only train and update model while AI is enabled
    if ai_enabled:
        agent.remember(
            state_tensor,
            action,
            reward,
            torch.tensor(np.array(next_state), dtype=torch.float32).to(DEVICE),
            terminated
        )
        agent.replay()

        if step % 1000 == 0:
            agent.update_target()

    # --- Debugging / Info ---
    acc = info.get("accuracy", None)
    acc_display = f"{acc:.2f}%" if acc is not None else "N/A"
    fps_display = info.get("fps", 0.0)
    score_display = info.get("score", 0)
    last_j = info.get("last_judgement", "-")
    judge_counts = info.get("judge_counts", {})

    print(f"\n=== Step {step} | FPS: {fps_display:.1f} ===")
    print(f"  🟣 Screen Non-Black: {song_active} | AI Enabled: {ai_enabled}")
    print(f"  Action: {action} | Reward: {reward:+.2f}")
    print(f"  Accuracy: {acc_display} | Score: {score_display:,} | Last: {last_j}")
    print(f"  Judges: 300g={judge_counts.get('300g',0)} | 300={judge_counts.get('300',0)} | 200={judge_counts.get('200',0)} | 100={judge_counts.get('100',0)} | 50={judge_counts.get('50',0)} | MISS={judge_counts.get('miss',0)}")

    if terminated:
        print("🎯 Result screen detected — saving model...")
        torch.save(agent.model.state_dict(), MODEL_PATH)
        break

    if truncated:
        print("⏱️ Episode limit reached — saving model...")
        torch.save(agent.model.state_dict(), MODEL_PATH)
        break

    # Move to next state
    state_tensor = torch.tensor(np.array(next_state), dtype=torch.float32).to(DEVICE)

# --- End of main loop ---
torch.save(agent.model.state_dict(), MODEL_PATH)
print("💾 Model saved. Session complete.")
env.close()
