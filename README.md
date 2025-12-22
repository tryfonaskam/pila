# 🚗 PILA — PolyTrack Imitation Learning Agent

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

## ✨ What Is PILA?

PILA learns how to play PolyTrack by:

- 🎥 Observing gameplay data (states + actions)
- 🧠 Training a neural network on this data
- 🎮 Reproducing player behavior in real time inside the game

This approach eliminates hand-written logic and relies entirely on **learning by example**.

---

## 🧠 How It Works

### 📊 Data Collection
- Gameplay is recorded as **observations (inputs)** and **actions(input and output)**
  
- Outputs represent player controls:
  - Steering
  - Throttle
  - Brake

##

### 🏋️ Training
- A neural network is trained using **imitation learning**
- The model minimizes loss between:
  - Predicted actions
  - Recorded player actions
- Trained models are saved as **checkpoints(every 2 epochs) or complete save** for later use
##

### ▶️ Inference / Playing
1. The trained model reads **live game frames**
2. It predicts the next actions using the current frame
3. Actions are sent to the game as **keyboard inputs**

---
## ⚙️ Requirements

-  Modern **CPU or GPU**
-  **Python 3.11.9**
---

## how to use PILA

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