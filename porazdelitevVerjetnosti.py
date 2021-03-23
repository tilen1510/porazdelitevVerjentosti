import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import module_statistics as stat
from scipy.interpolate import make_interp_spline, BSpline

data = pd.read_table("RV_vaja01_27.txt", header=None).values[:, 1]

t_max = np.max(data)  # najvišja vrednost
t_min = np.min(data)  # najnižja vrednost

m = 10  # število intervalov
N = len(data)  # število podatkov

delta_t = (t_max - t_min) / m + 0.00001  # velikost podintervala

t = np.arange(t_min + delta_t * 0.5, t_max + delta_t * 0.5, delta_t)    # srednje vrednosti intervalov

# na podlagi histograma je treba določiti za kakšno porazdelitev gre
print("Iz histograma razberi za kakšno porazdelitev gre.")
print("Vpiši črko: \n"
      "N - normalna porazdelitev \n"
      "W - Weibullova porazdelitev \n"
      "E - eksponentna porazdelitev")

# izris histograma
fig = plt.figure()
ax = fig.add_subplot()
freq, bins, patches = ax.hist(data, bins=m, density=False, color='darkred', rwidth=0.7, alpha=0.8, label='Histogram')
ax.set_xticks(np.arange(t[0], t[-1] + delta_t / 100, delta_t))
ax.set_xlabel('t')
ax.set_ylabel('Frekvenca')
ax.grid()
plt.show()

dist_type = input()     # vzame input, ki more biti ena od črk W, E ali N!

if dist_type == "N":    # iz modula vzame tisto metodo, ki jo želimo
    f, F = stat.norm_dist(x=data, t=t, print_values=True)
elif dist_type == "E":
    f, F = stat.exp_dist(x=data, t=t, print_values=True)
elif dist_type == "W":
    f, F = stat.Weibull_dist(x=data, t=t, print_values=True)
else:
    print("Nepravilen vnos!")


fig = plt.figure()
ax1 = fig.add_subplot()
ax2 = ax1.twinx()           # da riše plot na isti graf (potrebno zaradi seknudarne y-osi)

font = 13
t_new = np.linspace(t[0], t[-1], m * 10)

ax1.hist(data, bins=m, density=True, color='gold', rwidth=0.7, alpha=0.8)   # density = True -> relativna, False -> absolutna frekvenca
f_spl = make_interp_spline(t, f, k=2)  # type: BSpline                      # za glajenje krivulje
f_smooth = f_spl(t_new)
ax1.plot(t_new, f_smooth, label='f(t)', alpha=1, color="black", linewidth=2.5)
ax1.set_xticks(np.arange(t[0], t[-1] + delta_t / 100, delta_t))
ax1.set_xlabel('t', fontsize=font)
ax1.set_ylabel('f(t), fi*', fontsize=font)
ax1.grid()

F_spl = make_interp_spline(t, F, k=2)  # type: BSpline
F_smooth = F_spl(t_new)
ax2.plot(t_new, F_smooth, label='F(t)', linewidth=2.5, color='red')
ax2.set_yticks(np.arange(0, 1.1, 0.2))
ax2.set_ylabel('F(t)', fontsize=font)
ax2.set_ylim([0, 1.0])

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]      # da dobimo legendo za obe krivulji
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

fig.legend(lines, labels, bbox_to_anchor=(0.91, 0.75), fontsize=font+1)     # bbox_to_anchor določi pozivijo legende

plt.show()

fi = freq/N             # relativna frekvenca
fi_nor = fi/delta_t     # normirana relativna frekvenca

# print za izris tabele v Latex-u
print('\\begin{tabular}{cccccc} \n'
      '\\hline\n'
      't & frekvenca & $f_i$ & $f_i^*$ & $f(t)$ & $F(t)$ \\\\ \n '
      '\\hline\n'
      '\\hline')
for i in range(len(t)):
    print(f'{t[i]:.3f}'.replace('.', ','), '&', int(freq[i]), '&', f'{fi[i]:.3f}'.replace('.', ','), '&', f'{fi_nor[i]:.5f}'.replace('.', ','), '&', f'{f[i]:.5f}'.replace('.', ','), '&', f'{F[i]:.5f}'.replace('.', ','), '\\\\ \n ' 
          '\\hline')
print('\\end{tabular}')
