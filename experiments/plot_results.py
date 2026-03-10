import matplotlib.pyplot as plt

steps = []
losses = []

with open("experiments/run_001/logs.txt") as f:

    for line in f:

        step, loss = line.split(",")

        steps.append(int(step))
        losses.append(float(loss))


plt.plot(steps, losses)

plt.xlabel("Step")
plt.ylabel("Loss")

plt.title("Training Loss")

plt.show()