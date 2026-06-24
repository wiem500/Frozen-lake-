import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

GOAL_STATE = 63

def manhattan(s):
    return abs(GOAL_STATE//8 - s//8) + abs(GOAL_STATE%8 - s%8)

def shaped_reward(state, new_state, gym_reward, terminated):
    if gym_reward == 1:
        return 1.0
    if terminated and gym_reward == 0:
        return -0.5
    progres = (manhattan(state) - manhattan(new_state)) * 0.1
    return -0.01 + progres


def run(episodes, is_training=True, render=False):
    # ... (tout ton code actuel reste IDENTIQUE jusqu'à la fin de la fonction run)
    # Je ne change rien ici pour respecter ta demande

    env = gym.make(
        'FrozenLake-v1',
        map_name="8x8",
        is_slippery=True,
        render_mode='human' if render else None
    )

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('frozen_lake8x8.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a    = 0.1
    discount_factor_g  = 0.95
    epsilon            = 1.0
    epsilon_decay_rate = 0.0002
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)
    shaped_per_episode  = np.zeros(episodes)
    steps_per_episode   = np.zeros(episodes)
    success_count       = 0
    start_time          = time.time()

    print("=" * 75)
    print(f"  FrozenLake 8x8 | {'TRAINING' if is_training else 'TESTING'} | {episodes} épisodes")
    print(f"  lr={learning_rate_a} | gamma={discount_factor_g} | decay={epsilon_decay_rate}")
    print("=" * 75)
    print(f"  {'Episode':>8} | {'Result':>7} | {'Steps':>6} | {'R_shaped':>9} | {'Epsilon':>8} | {'WinRate%':>9} | {'Q_max':>7}")
    print("-" * 75)

    for i in range(episodes):
        state          = env.reset()[0]
        terminated     = False
        truncated      = False
        steps          = 0
        episode_shaped = 0.0

        while not terminated and not truncated:

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, gym_reward, terminated, truncated, _ = env.step(action)

            r_shaped = shaped_reward(state, new_state, gym_reward, terminated)
            episode_shaped += r_shaped

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    r_shaped
                    + discount_factor_g * np.max(q[new_state, :])
                    - q[state, action]
                )

            state  = new_state
            steps += 1

        rewards_per_episode[i] = gym_reward
        shaped_per_episode[i]  = episode_shaped
        steps_per_episode[i]   = steps

        if gym_reward == 1:
            success_count += 1

        epsilon = max(epsilon - epsilon_decay_rate, 0)
        if epsilon == 0:
            learning_rate_a = 0.00001

        if (i + 1) % 100 == 0 or i < 10:
            win_rate = (success_count / (i + 1)) * 100
            q_max    = np.max(q)
            status   = "✅ WIN " if gym_reward == 1 else "❌ FAIL"
            print(
                f"  {i+1:>8} | {status} | {steps:>6} | {episode_shaped:>+9.3f} | "
                f"{epsilon:>8.4f} | {win_rate:>8.1f}% | {q_max:>7.4f}"
            )

    elapsed  = time.time() - start_time
    final_wr = (success_count / episodes) * 100
    avg_steps = np.mean(steps_per_episode)
    avg_r     = np.mean(shaped_per_episode)

    print("=" * 75)
    print(f"  ✅ Terminé en {elapsed:.1f}s")
    print(f"  🏆 Taux de succès      : {final_wr:.1f}%  ({success_count}/{episodes})")
    print(f"  📊 Steps moyen         : {avg_steps:.1f}  (optimal ≈ 14)")
    print(f"  🎯 Reward shapée moy.  : {avg_r:+.3f}")
    print(f"  🔍 Epsilon final       : {epsilon:.4f}")
    print(f"  🧠 Q_max final         : {np.max(q):.4f}")
    print("=" * 75)

    env.close()
    _plot_metrics(rewards_per_episode, shaped_per_episode, steps_per_episode, episodes)

    if is_training:
        with open("frozen_lake8x8.pkl", "wb") as f:
            pickle.dump(q, f)
        print("  💾 Q-Table sauvegardée → frozen_lake8x8.pkl")


# ====================== NOUVELLE FONCTION DE TEST ======================
def test_agent(nb_episodes=10, render=True):
    """Fonction simple pour tester l'agent après entraînement"""
    print(f"\n🎮 Lancement du mode TEST sur {nb_episodes} épisodes avec affichage...")
    run(episodes=nb_episodes, is_training=False, render=render)


def _plot_metrics(rewards, shaped, steps, episodes):
    # (Ta fonction actuelle reste inchangée)
    window = 100

    win_rate  = np.array([np.mean(rewards[max(0,t-window):(t+1)])*100 for t in range(episodes)])
    avg_shaped= np.array([np.mean(shaped[max(0,t-window):(t+1)])      for t in range(episodes)])
    avg_steps = np.array([np.mean(steps[max(0,t-window):(t+1)])       for t in range(episodes)])

    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    fig.suptitle("FrozenLake 8x8 — Hyperparamètres optimisés", fontsize=14, fontweight='bold')

    axes[0].plot(avg_shaped, color='purple', linewidth=1)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='R=0 (seuil)')
    axes[0].fill_between(range(episodes), avg_shaped, 0, where=(avg_shaped >= 0), color='green', alpha=0.15, label='R positif ✅')
    axes[0].fill_between(range(episodes), avg_shaped, 0, where=(avg_shaped < 0),  color='red',   alpha=0.15, label='R négatif ❌')
    axes[0].set_title("Reward shapée moyenne — doit passer en positif")
    axes[0].set_ylabel("Reward shapée")
    axes[0].set_xlabel("Épisode")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(win_rate, color='green', linewidth=1)
    axes[1].axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50%')
    axes[1].axhline(y=75, color='green',  linestyle='--', alpha=0.5, label='75% (objectif)')
    axes[1].set_title("Taux de succès % — objectif : dépasser 75%")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].set_xlabel("Épisode")
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(avg_steps, color='coral', linewidth=1)
    axes[2].axhline(y=14, color='green', linestyle='--', alpha=0.7, label='Optimal ≈ 14 steps')
    axes[2].set_title("Steps moyen — doit baisser vers 14 (chemin optimal)")
    axes[2].set_ylabel("Steps")
    axes[2].set_xlabel("Épisode")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('frozen_lake8x8.png', dpi=150)
    plt.show()
    print("  📈 Graphes sauvegardés → frozen_lake8x8.png")


if __name__ == '__main__':
    # Entraînement
    run(30000, is_training=True, render=False)
    
    #=== Test après entraînement ===
    #test_agent(nb_episodes=10, render=True)