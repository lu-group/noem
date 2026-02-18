import matplotlib.pyplot as plt
# plt.figure(figsize=(4.4,4))
import numpy as np
import numpy as np

results = np.load("scalability_test_results.npy", allow_pickle=True)
model_acc_list = results.item().get("model_acc_list")
noe_acc_list = results.item().get("noe_acc_list")


print(model_acc_list)
print(noe_acc_list)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter, MaxNLocator

model_acc_list = np.array(model_acc_list)
noe_acc_list = np.array(noe_acc_list)
print(len(model_acc_list))

slope = 0.527
intercept = 0

fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=120)
ax.scatter(model_acc_list, noe_acc_list, color="blue", alpha=0.7)
ax.plot([-1, 1], [-slope, slope], color='black', lw=1.5, linestyle='--')

# Set x and y range

ax.set_xlim(0,  2.9e-3)
ax.set_ylim(0,  1.45e-3)
ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='both',  style='sci', scilimits=(-3, 3))
ax.xaxis.set_major_locator(MaxNLocator(4))
ax.yaxis.set_major_locator(MaxNLocator(4))

plt.tight_layout()
plt.show()

