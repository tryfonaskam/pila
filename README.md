# PILA (macOS) — Polytrack Imitation Learning Agent

**PILA (Polytrack Imitation Learning Agent)** is an experimental imitation learning system designed to learn and reproduce human gameplay behavior directly from raw interaction data. The project captures keyboard inputs and screen frames, converts them into a structured dataset, and trains a neural model capable of playing the game autonomously by imitating recorded human behavior.

This repository contains the **macOS implementation** of PILA, including:
- A data capture pipeline
- Dataset construction logic
- Model training workflow
- Real-time inference for autonomous gameplay

The long-term goal of PILA is to evolve into a **universal imitation learning framework** capable of learning arbitrary computer interactions (games, tools, workflows) without environment-specific hardcoding. That work is currently **in active development**.

---

## Demo

### Model Playing the Game (Short Clip)

Below is a short demo of the trained PILA model autonomously playing the game after learning from recorded human input:

![PILA Model Playing PolyTrack](assets/demo.gif)


### Full Demonstration Video (Extended)

For a longer, uninterrupted demonstration—including model stability and general behavior—watch the full video here:

**▶ Full demo video:**  
https://drive.google.com/file/d/1kZ5in9gRVzJvNwi8Tk2CXrAiEsIv1SD5/view?usp=sharing

---

## Project Overview

At a high level, PILA follows this pipeline:

### 1. Human Gameplay Recording
- Keyboard input is captured at high temporal resolution
- Screen frames are recorded and synchronized with inputs

### 2. Dataset Generation
- Input–frame pairs are serialized into a structured dataset
- Data is optimized for supervised imitation learning

### 3. Model Training
- A neural network is trained to predict actions from visual state
- Training is fully offline and reproducible

### 4. Autonomous Gameplay
- The trained model runs in real time
- Screen input → model inference → simulated keypresses

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sahusaurya/pila_mac.git
cd pila_mac
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Project (Recommended Order)

### Step 1: Capture Gameplay Data
Record your own gameplay to generate a dataset:
```bash
python -m capture.datasetmake
```

Follow the on-screen instructions to start and stop recording.

### Step 2: Train the Model
Train the imitation learning model on the generated dataset:
```bash
python train.py
```

Model checkpoints will be saved automatically.

### Step 3: Run the Trained Agent
Let the model play the game autonomously:
```bash
python play.py
```

---

## Pretrained Assets (For Quick Testing)

If you want to skip data collection and training, you can use the exact assets shown in the demo.

- **Sample dataset (Google Drive):** Raw screen frames and action labels collected from human PolyTrack gameplay (macOS).  
  https://drive.google.com/drive/folders/1f9Jg0l5n-Ip7juFT6VKr23beQFkOnazS?usp=sharing

Place it in:
```
datasets/
```
Preserve the internal structure and naming convention used inside the individual runs. The names of the directories of each recording session(runs) can be changed and will not affect the training or playing phases.

- **Pretrained model (.pt):** Trained on PolyTrack gameplay (macOS).  
  https://drive.google.com/file/d/15t0sMfw4ACcbCK3slHOAooJs1QdC0pGU/view

Place it in:
```
checkpoints/
```
The name of the model has to be pila_final.pt, since the path in play.py requires this.

---

## Platform Support

- **macOS:** This repository
- **Windows:** Separate implementation available here:  
  https://github.com/tryfonaskam/pila``/pila_windows

The two repositories share the same conceptual architecture but differ in OS-specific input capture and control logic.

---

## Current Status & Future Work

PILA is an **active research prototype**. Ongoing and planned improvements include:
- Universal, configurable key-mapping (no hardcoded controls)
- Mouse movement and click imitation
- Higher-frequency temporal modeling
- Cross-application generalization
- Modular environment abstraction

The broader objective is to develop PILA into a **general-purpose imitation learning engine** capable of learning arbitrary computer interactions directly from demonstrations.

---

## Credits

### Saurya Aditya Sahu  
**[sahusaurya](https://github.com/sahusaurya) — Project Co-Designer & macOS / Apple Silicon Lead**

Co-designed the project direction and independently designed and implemented the complete macOS- and Apple Silicon–specific PILA pipeline. Led the rewrite of the data capture system for macOS, including a modular screen capture and input-state architecture, canonical 512×512 preprocessing enforced consistently across data collection, training, and inference, and a macOS-compatible real-time control loop.

Additionally implemented Apple Silicon (MPS)–based training support, authored independent `train.py` and `play.py` pipelines aligned with the macOS capture format, and conducted standalone model training and experimentation on macOS systems. Contributed to early project direction by identifying PolyTrack as a suitable imitation-learning environment and assisting with test environment setup.

---

### Tryfonas Kampouropoulos  
**[tryfonaskam](https://github.com/tryfonaskam) — Project Author & Core Pipeline Lead**

Project author and lead developer. Designed and implemented the original PILA imitation learning pipeline, including the initial data capture logic, dataset organization, neural network architecture, training workflow, checkpointing, and real-time inference. Led early experimentation, baseline model training, documentation, and overall execution of the original system.

