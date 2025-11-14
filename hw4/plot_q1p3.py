import numpy as np
import matplotlib.pyplot as plt
import os
# import imageio
import glob


os.makedirs("hw4/plots", exist_ok=True)



ep_len = np.loadtxt("data/episode_len.csv", delimiter=",").flatten()
n = np.linspace(0, ep_len.size, ep_len.size)

plt.plot(n, ep_len)
plt.xlabel("episode")
plt.ylabel("# steps for convergence")
plt.grid()

plt.title("# convergence steps vs episode")
plt.tight_layout()
plt.savefig("hw4/plots/ep_len.pdf")
plt.close()


tot_reward = np.loadtxt("data/total_reward.csv", delimiter=",").flatten()
n = np.linspace(0, tot_reward.size, tot_reward.size)

plt.plot(n, tot_reward)

plt.xlabel("episode")
plt.ylabel("total reward")
plt.grid()

plt.title("total reward vs episode")
plt.tight_layout()
plt.savefig("hw4/plots/tot_reward.pdf")
plt.close()










qtable_files = glob.glob("data/qtable_[0-9]*.csv")


max_ = -np.inf
min_ = np.inf
for f in qtable_files:
    Q = np.loadtxt(f, delimiter=",")
    if np.max(Q) > max_:
        max_ = np.max(Q)
    if np.min(Q) < min_:
        min_ = np.min(Q)
        

for f in qtable_files:
    Q = np.loadtxt(f, delimiter=",")
    plt.imshow(Q, interpolation="nearest", vmin=min_, vmax=max_)
    plt.colorbar()

    episode = f.removeprefix("data/qtable_").removesuffix(".csv")

    plt.title(f"episode: {episode}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(f"hw4/plots/qtable_{episode}.pdf")
    plt.close()



