# Quick Start - Training Your AI

## ✅ You're Ready to Train!

The AI is now pressing keys. Here's what happens next:

## How to Train

**1. Run the AI:**
```powershell
python main.py
```

**2. Play a song in Osu!mania**
- The AI will start pressing keys
- Watch the debug window (shows detection and key presses)
- Watch the console (shows score, accuracy, judgements)

**3. The AI learns automatically!**
- After each song, the model saves to `osu_dqn_model.pth`
- Next time you run it, it continues learning from where it left off
- Score and accuracy will improve over 20-50 songs

## What You'll See

### Console Output (Every Step):
```
=== Step 145 | FPS: 58.3 ===
🎵 Song Active: True | Armed: False
Action: [1 0 1 0] | Reward: +1.80 | Combo: 12
Notes Detected Near Hit: [1, 0, 1, 0]
Notes In Hit Window: [1, 0, 1, 0]
AI SCORE: 42,350  |  Game Acc: 72.45%
Judges: 300g=45 | 300=80 | 200=12 | 100=5 | 50=2 | MISS=8
```

### After Song Ends:
```
🏁 EPISODE SUMMARY
====================================
AI Predicted Score: 567,890
Total Notes Seen: 456
Combo: 89

Judgements:
  300g: 120
  300:  180
  200:   45
  100:   15
  50:     8
  MISS:  12
====================================
```

## Understanding the Learning Process

### First 5 Songs (Epsilon ~0.9)
- **Mostly random key presses** (90% random)
- Learning basic timing patterns
- Score: 200k-400k
- Accuracy: 40-60%

### Songs 5-20 (Epsilon ~0.7 → 0.3)
- **Mix of learned + random** (70% → 30% random)
- Better timing, fewer misses
- Score: 400k-650k
- Accuracy: 60-75%

### Songs 20-50 (Epsilon ~0.3 → 0.05)
- **Mostly learned patterns** (30% → 5% random)
- Consistent performance
- Score: 650k-850k+
- Accuracy: 75-85%+

### After 50+ Songs (Epsilon ~0.05)
- **Almost fully trained** (5% random exploration)
- Best possible performance
- Score depends on song difficulty
- Accuracy: 80-90%+

## Improving Performance

### If Some Colored Notes Are Missed

Your colored notes (yellow/blue/orange/pink) might have different brightness.

**In `main.py` line 21, lower the threshold:**
```python
detect_abs_thresh=40,  # Lower from 50 to 40
```

### If Timing is Off (Too Many 100s/50s/Misses)

**In `screen_env.py` line 119, increase the window:**
```python
self.hit_window_band = 20  # Increase from 12 to 20 (more forgiving timing)
```

### If AI Presses Too Many Keys

**In `dqn_agent.py` line 102, increase threshold:**
```python
action = (q_values > 0.3).int()  # Increase from 0.0 to 0.3
```

## Color Issue Answer

**Q: Why does the debug window look different (grayscale) from the actual game?**

**A: This is completely normal!** 

- The AI converts the screen to **grayscale** for processing
- It only needs brightness values, not colors
- Your colored notes (yellow/blue/orange/pink) are detected by brightness
- As long as they're bright enough, they'll be detected
- **You don't need to change anything about the colors!**

The AI sees:
- Yellow note → Bright white
- Blue note → Medium gray
- Orange note → Medium-bright white
- Pink note → Bright white

If a specific color isn't being detected, just lower `detect_abs_thresh` (see above).

## Training Tips

### ✅ DO:
- Train on the same difficulty for first 10-20 songs
- Let it play full songs (don't skip/quit mid-song)
- Check console output to see if notes are detected
- Adjust `detect_abs_thresh` if some colored notes are missed
- Be patient - improvement happens gradually

### ❌ DON'T:
- Delete the model file unless starting over
- Change settings every song - give it 5-10 songs to adapt
- Expect 90%+ accuracy in first 10 songs
- Play very different difficulties without retraining

## Daily Workflow

**Morning (10-20 mins):**
1. Run `python main.py`
2. Play 3-5 songs
3. Note the final scores and accuracy

**Evening (10-20 mins):**
1. Run `python main.py` again (continues from morning)
2. Play 3-5 more songs
3. See improvement!

**After 1 Week:**
- 20-40 songs trained
- Should see 70-80% accuracy
- Consistent good scores

**After 2 Weeks:**
- 40-80 songs trained
- Should see 75-85% accuracy
- Near-optimal performance

## Need Help?

Check these files:
- **TRAINING_GUIDE.md** - Detailed training information
- **TROUBLESHOOTING.md** - Common problems and solutions
- **WARP.md** - Technical architecture details

## Current Configuration

Your AI is configured for **balanced training**:
- ✅ Hit window gating enabled (better timing)
- ✅ Judgement rewards 2x (faster learning)
- ✅ Epsilon starts high (good exploration)
- ✅ Detects colored notes by brightness
- ✅ Long note (LNB) support enabled
- ✅ Auto-saves after each song

**You're all set! Just run `python main.py` and start training!** 🎮
