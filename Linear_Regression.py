import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

brown_area_samples = np.array([0, 2.1, 2.43, 10.05])  # Example brown area values
gradient_samples = np.array([0, 0.022, 0.05, 0.2])  # Corresponding gradients values

data = {"Brown Area Samples": brown_area_samples, "Gradient Samples": gradient_samples}
df = pd.DataFrame(data)

# Plot using seaborn
plt.figure(figsize=(8, 6))
sns.regplot(x=brown_area_samples, y=gradient_samples, marker='o')
plt.title("Brown Area Samples vs. Gradient Samples")
plt.xlabel("Brown Area Samples")
plt.ylabel("Gradient Samples")
plt.grid(True)
plt.show()