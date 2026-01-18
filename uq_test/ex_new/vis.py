import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

op_csv = Path("operator_learning_results.csv")
noem_csv = Path("noem_learning_results.csv")
fno_csv = Path("fno_learning_results.csv")

df_op = pd.read_csv(op_csv)    # columns: L, Sample_num, Error
df_no = pd.read_csv(noem_csv)  # columns: L, Sample_num, Error
df_fno = pd.read_csv(fno_csv)  # columns: L, Sample_num, Error


noem_data = {}
for _, row in df_no.iterrows():
    L = row['L']
    error = row['Error']
    noem_data[L] = error
print(noem_data)

n_list = [1,5,10,15,20, 25, 30,35,40,45,50]
noem_results_avg = []
noem_results_std = []
for n in n_list:
    # Compute the average error for L <= n
    errors = [noem_data[L] for L in noem_data if L <= n]
    avg_error = np.mean(errors) if errors else None
    noem_results_avg.append(avg_error)
    std_error = np.std(errors) if errors else None
    noem_results_std.append(std_error)

total_sam_num = 2000
no_results_avg = []
no_results_std = []
for n in n_list:
    sam_num = int(total_sam_num / n)
    errors = df_op[(df_op['Sample_num'] == sam_num)]['Error'].values
    avg_error = np.mean(errors)
    no_results_avg.append(avg_error)
    std_error = np.std(errors)
    no_results_std.append(std_error)

fno_results_avg = []
fno_results_std = []
for n in n_list:
    sam_num = int(total_sam_num / n)
    errors = df_fno[(df_fno['Sample_num'] == sam_num)]['Error'].values
    avg_error = np.mean(errors)
    fno_results_avg.append(avg_error)
    std_error = np.std(errors)
    fno_results_std.append(std_error)

plt.figure(figsize=(10, 6))
# Set all the font size to 16
plt.rcParams.update({'font.size': 16})
plt.errorbar(n_list, no_results_avg, yerr=no_results_std, label='DeepONet', marker='o', capsize=5, linestyle='--')
plt.errorbar(n_list, fno_results_avg, yerr=fno_results_std, label='FNO', marker='s', capsize=5, linestyle='--')
plt.errorbar(n_list, noem_results_avg, yerr=noem_results_std, label='NOEM', marker='^', capsize=5, linestyle='--')
plt.yscale('log')
plt.xlabel('Number of systems, $N$')
plt.ylabel('Relative $L^2$ error')
# y range from 1e-2 to 1
# plt.ylim(1e-2, 1)
# x range from 0 to 21
plt.xlim(0, 51)
plt.xticks(n_list)
plt.legend()
plt.legend(frameon=False, fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig("noem_vs_deeponet.png", dpi=
300)
plt.show()

