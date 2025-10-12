# 🧩 Guida al verbose di training (es. PPO Reinforcement Learning)

Durante l’addestramento, il terminale mostra un log (“verbose”) come questo:

```text
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 245         |    Lunghezza media di un episodio (in passi).
|    ep_rew_mean          | -32.4       |    Ricompensa media per episodio.
| time/                   |             |
|    fps                  | 1214        |    Frame per secondo.
|    iterations           | 11          |    Numero di iterazioni completate.
|    time_elapsed         | 148         |    Tempo totale trascorso (in secondi).
|    total_timesteps      | 180224      |    Numero totale di step simulati.
| train/                  |             |
|    approx_kl            | 0.005239252 |    Distanza KL tra la policy vecchia e la nuova.
|    clip_fraction        | 0.0376      |    Percentuale di aggiornamenti “clippati”.
|    clip_range           | 0.2         |    Range massimo del clipping.
|    entropy_loss         | -2.85       |    Entropia della policy.
|    explained_variance   | 0.889       |    Varianza spiegata dalla value function.
|    learning_rate        | 0.0003      |    Tasso di apprendimento.
|    loss                 | -0.0206     |    Perdita totale.
|    n_updates            | 100         |    Numero totale di aggiornamenti del modello.
|    policy_gradient_loss | -0.00393    |    Perdita associata alla policy.
|    std                  | 1.01        |    Deviazione standard delle azioni.
|    value_loss           | 0.0356      |    Errore nella stima della funzione di valore.
-----------------------------------------
```

---

### Approfondimenti
Metriche relative all’ottimizzazione della rete neurale.

- **approx_kl**: se cresce troppo c'è rischio di aggiornamenti instabili.  
- **clip_fraction**: se troppo alta il clipping limita eccessivamente l’apprendimento.  
- **clip_range**: tipicamente tra 0.1 e 0.3.  
- **entropy_loss**: valore basso implica una policy più deterministica, alto più esplorazione.  
- **explained_variance**: vicino ad 1 implica buone stime del valore, vicino a 0 stime scarse.
- **policy_gradient_loss**: valori piccoli indicano aggiornamenti più stabili.
- **std**: se alta indica esplorazione, se bassa ottimizzazione.
- **value_loss**: più basso indica una value network più accurata.

---

## Consigli pratici

- **Performance**: controlla `ep_rew_mean` e `ep_len_mean`. Devono generalmente aumentare nel tempo.
- **Stabilità**: osserva `approx_kl` e `clip_fraction`.  
  - Se `approx_kl` cresce troppo, riduci `learning_rate`.  
  - Se `clip_fraction` è > 0.3, il clipping limita troppo gli aggiornamenti.  
- **Value function**: `explained_variance` vicino a **1.0** = buone previsioni del valore.  
- **Esplorazione**: se `entropy_loss` cala rapidamente, la policy si irrigidisce troppo (rischio di overfit).

---
