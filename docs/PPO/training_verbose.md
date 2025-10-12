# 🧩 Guida al verbose di training (es. PPO Reinforcement Learning)

Durante l’addestramento, il terminale mostra un log (“verbose”) come questo:

-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 245         |
|    ep_rew_mean          | -32.4       |
| time/                   |             |
|    fps                  | 1214        |
|    iterations           | 11          |
|    time_elapsed         | 148         |
|    total_timesteps      | 180224      |
| train/                  |             |
|    approx_kl            | 0.005239252 |
|    clip_fraction        | 0.0376      |
|    clip_range           | 0.2         |
|    entropy_loss         | -2.85       |
|    explained_variance   | 0.889       |
|    learning_rate        | 0.0003      |
|    loss                 | -0.0206     |
|    n_updates            | 100         |
|    policy_gradient_loss | -0.00393    |
|    std                  | 1.01        |
|    value_loss           | 0.0356      |
-----------------------------------------

---

## 📊 Sezioni principali

### rollout/
Statistiche raccolte durante l’interazione dell’agente con l’ambiente.

- **ep_len_mean** → Lunghezza media di un episodio (in passi).  
  Valore più alto = l’agente agisce o sopravvive più a lungo.  
- **ep_rew_mean** → Ricompensa media per episodio.  
  Valore più alto (o meno negativo) = performance migliore.

---

### time/
Informazioni sul tempo e sullo stato di avanzamento del training.

- **fps** → Frame per secondo. Indica la velocità del training.  
- **iterations** → Numero di iterazioni completate.  
- **time_elapsed** → Tempo totale trascorso (in secondi).  
- **total_timesteps** → Numero totale di step simulati.

---

### train/
Metriche relative all’ottimizzazione della rete neurale.

- **approx_kl** → Distanza KL approssimata tra la policy vecchia e la nuova.  
  Se cresce troppo → rischio di aggiornamenti instabili.  
- **clip_fraction** → Percentuale di aggiornamenti “clippati”.  
  Se troppo alta → il clipping limita eccessivamente l’apprendimento.  
- **clip_range** → Range massimo del clipping (parametro PPO).  
  Tipico: 0.1–0.3.  
- **entropy_loss** → Entropia della policy.  
  Valore basso → policy più deterministica; alto → più esplorazione.  
- **explained_variance** → Varianza spiegata dalla value function.  
  Vicino a 1 → buone stime del valore; vicino a 0 → stime scarse.  
- **learning_rate** → Tasso di apprendimento.  
- **loss** → Perdita totale.  
  Da interpretare in base all’implementazione: non sempre “più basso = meglio”.  
- **n_updates** → Numero totale di aggiornamenti del modello.  
- **policy_gradient_loss** → Perdita associata alla policy.  
  Valori piccoli → aggiornamenti più stabili.  
- **std** → Deviazione standard delle azioni (livello di esplorazione).  
  Alta = esplorazione; bassa = ottimizzazione.  
- **value_loss** → Errore nella stima della funzione di valore (V(s)).  
  Più basso → value network più accurato.

---

## 📈 Come leggerlo “al volo”

- **Performance** → controlla `ep_rew_mean` e `ep_len_mean`.  
  Devono generalmente aumentare nel tempo.  
- **Stabilità** → osserva `approx_kl` e `clip_fraction`.  
  - Se `approx_kl` cresce troppo, riduci `learning_rate`.  
  - Se `clip_fraction` è > 0.3, il clipping limita troppo gli aggiornamenti.  
- **Value function** → `explained_variance` vicino a **1.0** = buone previsioni del valore.  
- **Esplorazione** → se `entropy_loss` cala rapidamente, la policy si irrigidisce troppo (rischio di overfit).

---