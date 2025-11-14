# OsuManiaDQN

An AI agent that learns to play Osu!mania using Deep Q-Networks. This project contains the original scripts and assets migrated into a clean repository structure.

Contents
- dqn_agent.py: DQN agent logic
- FrameStack.py: Frame stacking/preprocessing utilities
- screen_env.py: Environment wrapper for screen capture and key presses
- main.py: Entry point / training or inference loop
- zones.json: Hit-zone configuration (duplicated under config/zones.json)
- osu_dqn_model.pth: Trained model weights

Notes on skins and hit zones
- Different skins may require different hit-zone coordinates. Adjust zones.json (or config/zones.json) to match your skin and resolution.
- If notes are visually detected but not being hit, verify: (a) zone alignment with your skin, (b) key mapping used for key presses, and (c) timing offset.

Getting started
- Ensure Python 3.9+ and GPU-enabled PyTorch if available.
- Install dependencies (to be defined) and run main.py.
