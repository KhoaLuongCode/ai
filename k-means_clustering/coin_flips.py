import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Observations: head, head, tail, head
# Represent heads as 1 and tails as 0
observations = [1, 1, 0, 1]

# Initial prior parameters for the Beta distribution (uniform prior)
alpha_prior = 1
beta_prior = 1

# Initialize lists to store alpha and beta parameters after each observation
alpha_list = [alpha_prior]
beta_list = [beta_prior]

# Update parameters after each observation
for obs in observations:
    if obs == 1:  # Head observed
        alpha_prior += 1
    else:         # Tail observed
        beta_prior += 1
    alpha_list.append(alpha_prior)
    beta_list.append(beta_prior)

# Theta values for plotting
theta = np.linspace(0, 1, 500)

# Plotting the posterior distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i in range(4):
    alpha = alpha_list[i+1]
    beta_param = beta_list[i+1]
    posterior = beta.pdf(theta, alpha, beta_param)
    
    axes[i].plot(theta, posterior, color='blue')
    axes[i].fill_between(theta, posterior, color='lightblue', alpha=0.5)
    axes[i].set_title(f'Posterior after {i+1} flip(s)')
    axes[i].set_xlabel(r'$\theta$')
    axes[i].set_ylabel(r'$p(\theta|D)$')
    axes[i].set_xlim(0, 1)
    axes[i].set_ylim(0, np.max(posterior) + 1)

    axes[i].annotate(f'α = {alpha}\nβ = {beta_param}', xy=(0.7, np.max(posterior)/2))

plt.tight_layout()
plt.show()
