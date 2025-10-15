## Training examples

<p align="center">
  <img src="/python/images/esempio_1.gif" alt="Esempio 1" width="30%">
  <img src="/python/images/esempio_2.gif" alt="Esempio 2" width="30%">
  <img src="/python/images/esempio_3.gif" alt="Esempio 3" width="30%">
</p>

## ⚙️ Installation & Setup

### 🧩 Prerequisites
- Python 3.10.12

### 💻 Installation
```
git clone https://github.com/pcestola/ParkingPPO.git
cd ParkingPPO/python
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 🏋️ Training (PPO)

Run the PPO training script from inside the python folder:

```
python training/train_ppo.py
```

Trained episode replays (.json files) will be automatically saved in:

```
ParkingPPO/python/replays/
```

## 🎬 Visualizing Replays

To visualize a replay (animated with Matplotlib):

```
python training/view_json.py <--file path/to/replay.json> <--index n> <--interval ms> <--tail frames> <--speed factor> <--gif>
```
