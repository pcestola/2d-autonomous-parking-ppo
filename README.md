# godot-ppo-parking (Godot 4.4.1, GDScript 4)

Progetto completo per un **ambiente di parcheggio** con:
- **Demo 3D WASD** (scena pronta `Main.tscn` con `Car3D`).
- **PPO headless** già impostato (MLP + GAE + clipping), con loop di training e valutazione.

## Requisiti
- **Godot 4.4.1**.
- Nessuna dipendenza esterna.

## Struttura
godot-ppo-parking/
├─ project.godot
├─ .gitignore
├─ LICENSE
├─ README.md
├─ scenes/
│ ├─ Main.tscn
│ └─ Car3D.tscn
├─ scripts/
│ ├─ utils/
│ │ └─ logger.gd
│ ├─ physics/
│ │ ├─ car_physics.gd
│ │ ├─ simple_ackermann_physics.gd
│ │ └─ bicycle_model_physics.gd
│ ├─ car/
│ │ ├─ car3d.gd
│ │ └─ car_virtual.gd
│ └─ rl/
│ ├─ mlp.gd
│ ├─ ppo.gd
│ └─ trainer_demo.gd

## Demo 3D (WASD)
1. Apri il progetto in **Godot 4.4.1**.
2. Apri `scenes/Main.tscn` e premi **Play**.
   - W/S avanti/indietro, A/D sterzo (A=sinistra, D=destra), R reset.
   - L'auto parte verso **-Z** (va via dalla camera con W).

## Training PPO (headless)
Terminale → nella cartella del progetto:
godot4 --headless --path . --script res://scripts/rl/trainer_demo.gd

### Parametri default (in `trainer_demo.gd`)
- `iters = 30`, `steps = 2048`, `batch = 256`, `epochs = 5`, `dt = 0.05`
