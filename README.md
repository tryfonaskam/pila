# PILA — PolyTrack Imitation Learning Agent

<p align="center">
  <a href="https://discord.gg/FqaBgZBGtG" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Discord-Join%20Server-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://www.tiktok.com/@tryfonaskam" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/TikTok-%23000000.svg?style=for-the-badge&logo=TikTok&logoColor=white" alt="License: Apache-2.0"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0.html" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/License-Apache2.0-blue.svg" alt="License: Apache-2.0"></a>
  
  
</p>

**PILA (PolyTrack Imitation Learning AI)** is an imitation learning agent trained to play **PolyTrack** by learning directly from recorded human gameplay — not from hard-coded rules.

Instead of manually programming behavior, PILA uses **supervised learning** to map game states to player actions, allowing it to imitate realistic human driving.

<div align='center'>
  your view of the imitation learning agent driving
  <br>
  <img alt="your view" src="https://github.com/tryfonaskam/pila/blob/main/gifs/original.gif">
</div>

<div align='center'>
  imitation learning agent driving view
  <br>
  <img alt="AGENT VIEW" src="https://github.com/tryfonaskam/pila/blob/main/gifs/agent_view.gif">
</div>

---

## What Is PILA?

PILA learns how to play PolyTrack by:

- Observing gameplay data (states + actions)
- Training a neural network on this data
- Reproducing player behavior in real time inside the game

This approach eliminates hand-written logic and relies entirely on **learning by example**.

---

## How It Works

### Data Collection
- Gameplay is recorded as **observations (inputs)** and **actions(input and output)**
  
- Outputs represent player controls:
  - Steering
  - Throttle
  - Brake

##

### Training
- A neural network is trained using **imitation learning**
- The model minimizes loss between:
  - Predicted actions
  - Recorded player actions
- Trained models are saved as **checkpoints(every 2 epochs) or complete save** for later use
##

### Inference / Playing
1. The trained model reads **live game frames**
2. It predicts the next actions using the current frame
3. Actions are sent to the game as **keyboard inputs**

---
## Requirements

-  Modern **CPU or GPU**
-  **Python 3.11.9**
---

## how to use PILA
**for macOS visit the [macOS](https://github.com/tryfonaskam/pila/tree/macOS) branch**

**for windows and linux follow this guide⬇**
### start by cloning the repo

```bash
git clone https://github.com/tryfonaskam/pila.git
cd pila
```
after you successfully cloned the repo you will need to install the requirements for the code to work correctly to do that run this command

```bash
pip -r requirements.txt
```

or install the manually
> note all requirements should be installed
> 
the installation is now complete.

---

## dataset creation and training

you should start with one of these commands

***⬇for automatic recapture⬇***

```bash
python3 loop_datamaker.py
```
***⬇you have to re-run the script each run⬇***

```bash
python3 datasetmake.py
```

after the execution of ***ONE*** of the scripts you should switch to your game window

>the game window should be windowed fullscreen for the code to work automatically


>or you ***NEED*** to set REGION = [None = full monitor] or set (left, top, right, bottom)

press ***F1*** to start capturing. then play the game and press ***F10*** to stop and save
> NOTE 1. you need to play the game smoothly and correctly for the model to be good

> NOTE 2. if you run ```python3 datasetmake.py``` you will need to re-start the script
> if you run ```python3 loop_datamaker.py``` you can just press ***F1*** to start capturing again

***the keys that are geting captured are just  ```Shift```, ```Ctrl```, ```w```, ```a```, ```s```, ```d```, ```q```, ```e``` and ```mouse movment```***

>[!TIP]
> increase the FPS for better capturing

>[!CAUTION]
>this will greatly increase the size of each run

---

## training

for this step you will need to have a modern GPU or CPU
to start training run this command⬇
```bash
python3 train.py
```
this will start training of the model using the datasets/ directory.
a checkpoint is saved every SAVE_EVERY = N the default is 2 in the checkpoints/ directory
you can stop training at any point and re-run the script to automatically resume training

to see a graphical representation of the loss, mag_key and mag_mouse run this command⬇

**for linux⬇**
```bash
tensorboard --bind_all --logdir runs/
```
for windows you can run the ```tensorboard.bat``` file or run this command⬆
# training settings

```bash
BASE_WEIGHT = 0.4             # minimum weight for zero frames
DYNAMIC_WEIGHT = 0.1          # how much nonzero frames add (0–1)
ATTN_SCALE = 0.1              # sensitivity
```

>NOTE: a nonzero frame is a frame that has a value of 0 inside the actions.csv

>```BASE_WEIGHT```minimum importance of every frame. if a frame is 0 It still contributes 40% of a normal loss

>```DYNAMIC_WEIGHT```Extra weight added when something happens. The more action → the more weight (up to a limit).

>```ATTN_SCALE = 0.1```ATTN_SCALE = 0.1. Controls how fast attention grows

---

## playing

to get the model to play the game you need to run this command⬇

```bash
python3 play.py
```

after that you have ~5 seconds to switch to your game
>NOTE: it needs to be the same size and resolution as in training

now the model will play the game

>[!IMPORTANT]
>if anything goes wrong press "esc" to stop the model

## advantages

one big list of advantages is the User-Friendly and Robust Workflow

***1. Automatic Data Organization: The data collection scripts automatically create and manage datasets in clearly labeled folders (run01, run02, etc.), keeping your experiments organized.***

***2. Resumable Training: You can stop and resume the training process at any time without losing progress, which is perfect for long training sessions and serious projects.***

***3. Integrated Visualization: With TensorBoard support, you can easily graph the model's learning progress to see how its performance improves over time.***

***Flexible and Configurable: Key parameters for data collection, training, and inference are clearly defined, making it easy to experiment and tune the agent's behavior.***

***Clear and Modular Codebase: The project is well-structured and serves as an excellent learning resource for anyone interested in imitation learning, computer vision, or game AI.***

## ***KEY ADVANTAGES***
***1. Another advantage is that the code’s modular, flexible design allows it to be reused across different games and applications. This adaptability makes it easy to integrate into new projects or expand its functionality without rewriting the core logic.***

***2. End-to-End Imitation Learning: the project provides a complete, self-contained pipeline for building a game-playing AI. It handles everything from data collection and training to real-time inference, making it a comprehensive solution.***

***3. Learns by Example, Not by Rules: PILA learns directly from observing human gameplay. This "show, don't tell" approach is powerful because it requires no hard-coded logic or manual programming of behaviors, allowing it to  learn complex and nuanced driving styles.***

***4. Real-Time Performance: The agent operates in real-time, using efficient screen capturing (dxcam) and a streamlined model to react to live gameplay without significant lag.***

***5. Intelligent Training with Attention: The training process uses a custom attention mechanism that gives more weight to frames with significant player actions. This helps the model focus on the most important moments of gameplay, leading to more efficient and effective learning.***

## **extra stuff**

**if you would like to have more/less inputs and outputs you can do that**
## loop_datamaker
**setp 1. add/remove keys inside loop_datamaker.py**

```bash
# Control keys
KEY_W = 'w'
KEY_S = 's'
KEY_A = 'a'
KEY_D = 'd'
KEY_Q = 'q'
KEY_E = 'e'
KEY_SPACE = 'space'
KEY_SHIFT = 'shift'
KEY_CTRL = 'ctrl'

# mouse buttons
MOUSE_LEFT = "left"
MOUSE_RIGHT = "right"
```

**step 2. then you will need to update the code to use the new keys**
```bash
# Keyboard controls
w_s = axis(KEY_S, KEY_W)
a_d = axis(KEY_A, KEY_D)
q_e = axis(KEY_Q, KEY_E)
space = 1.0 if keyboard.is_pressed(KEY_SPACE) else 0.0
shift_ctrl_val = shift_ctrl()

# mouse controls
left_click  = 1.0 if mouse.is_pressed(MOUSE_LEFT) else 0.0
right_click = 1.0 if mouse.is_pressed(MOUSE_RIGHT) else 0.0
```
**step 3. update this part**

```bash
# Save record
records.append([
    frame_name,
    w_s,
    a_d,
    q_e,
    space,
    shift_ctrl_val,
    mouse_dx_scaled,
    mouse_dy_scaled,
    left_click,
    right_click
])
```

**final. step 4. update this**
```bash
# Save data after stopping
df = pd.DataFrame(records, columns=[
    "frame","w_s","a_d","q_e","space","shift_ctrl","mouse_dx","mouse_dy","left_click","right_click"
])
```
## train
**step 1. update this part by adding or removing keys(should be the same as in loop_datamaker.py)**
```bash
y = np.array([
    float(rows[i]["w_s"]),
    float(rows[i]["a_d"]),
    float(rows[i]["q_e"]),
    float(rows[i]["space"]),
    float(rows[i]["shift_ctrl"]),
    float(rows[i]["mouse_dx"]),
    float(rows[i]["mouse_dy"]),
    float(rows[i]["left_click"]),
    float(rows[i]["right_click"])
], dtype=np.float32)
```
**final. step 2. change so the last numbere is the number of outputs from step. 1⬆**
```bash
class ControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 9) #the last numbere is the number of outputs
```

## play

**step 1. you can copy and paste step 2⬆ to  play.py do that here**

```bash
class ControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 9)
```
**step 2. update this code with you new inputs/outputs**
```bash
w_s, a_d, q_e, space, shift_ctrl, mouse_dx, mouse_dy, left_click, right_click = out
```

**step 3. update these accordingly**
```bash
 # W and S
    if w_s > WASD_THRESH:
        keyboard.press("w")
        keyboard.release("s")
    elif w_s < -WASD_THRESH:
        keyboard.press("s")
        keyboard.release("w")
    else:
        keyboard.release("w")
        keyboard.release("s")

    # A and D
    if a_d > WASD_THRESH:
        keyboard.press("d")
        keyboard.release("a")
    elif a_d < -WASD_THRESH:
        keyboard.press("a")
        keyboard.release("d")
    else:
        keyboard.release("a")
        keyboard.release("d")


    # Q and E
    if q_e > QE_THRESH:
        keyboard.press("e")
        keyboard.release("q")
    elif q_e < -QE_THRESH:
        keyboard.press("q")
        keyboard.release("e")
    else:
        keyboard.release("q")
        keyboard.release("e")

    # space and mouse clicks
    if space > SPACE_THRESH:
        keyboard.press("space")
    else:
        keyboard.release("space")
    
    if left_click > CLICK_THRESH:
        mouse.press(button='left')
        mouse.release(button='right')
    elif right_click < -CLICK_THRESH:
        mouse.press(button='right')
        mouse.release(button='left')
    else:
        mouse.release(button='left')
        mouse.release(button='right')
```

**final update this**
```bash
# stop release all keys
camera.stop()
keyboard.release("shift")
keyboard.release("ctrl")
keyboard.release("q")
keyboard.release("e")
keyboard.release("w")
keyboard.release("a")
keyboard.release("s")
keyboard.release("d")
keyboard.release("space")
mouse.release(button='left')
mouse.release(button='right')
```

## exetra 

you can run ```bash
python3 cluster.py```

to visualize patterns in the training data.

<div style="text-align:left;">
  <p>it is going to look like this</p>
  <img src="https://github.com/tryfonaskam/pila/blob/main/gifs/patterns.png" alt="pattern view" style="max-width:50%; height:auto;">
</div>


---

## Credits

**[tryfonaskam](https://github.com/tryfonaskam)** - Project author and lead developer.  
Designed and implemented the full imitation learning pipeline, including data capture, dataset organization, neural network architecture, training workflow, checkpointing, and real-time inference, contributor for pila macOS. Responsible for model training, experimentation, documentation, and overall project execution and contributing a custom-designed test track used for evaluation.

**[sahusaurya](https://github.com/sahusaurya)** - Project ideation and environment contribution.  
Helped shape the initial project direction, identifying PolyTrack as an appropriate environment for imitation learning, and contributing a custom-designed test track used for evaluation.

**[polytrack](https://kodub.itch.io/polytrack)** - the game used for training the model, and play the model
