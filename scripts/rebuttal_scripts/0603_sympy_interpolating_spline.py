import pandas as pd
import numpy as np
import sympy
import matplotlib.pyplot as plt

# do an interpolation using sympy for EC glucose and add as a boundary condition
data = pd.read_csv('datasets/VanHeerden_Glucose_Pulse/FF1_timeseries_format.csv', index_col=0)
domain = [float(i) for i in data.loc['ECglucose'].dropna().index]
drange = data.loc['ECglucose'].dropna().values

t = sympy.Symbol('t')

spline_d1 = sympy.interpolating_spline(1, t, domain, drange)
l_spline_d1= sympy.lambdify(t, spline_d1)


spline_d2 = sympy.interpolating_spline(2, t, domain, drange)
l_spline_d2= sympy.lambdify(t, spline_d2)

spline_d3 = sympy.interpolating_spline(3, t, domain, drange)
l_spline_d3= sympy.lambdify(t, spline_d3)

spline_d4 = sympy.interpolating_spline(4, t, domain, drange)
l_spline_d4= sympy.lambdify(t, spline_d4)


t=np.linspace(0,400,400)

fig,ax = plt.subplots(figsize=(10,5))
ax.plot(t, l_spline_d1(t), label='d1')
ax.plot(t,l_spline_d2(t),label="d2")
ax.plot(t,l_spline_d3(t), label="d3")
ax.plot(t,l_spline_d4(t),label="d4")
plt.title("ECglucose interpolation for degrees 1-4")
ax.set_xlabel("Time (in seconds)")
ax.set_ylabel("ECglucose")
plt.scatter(domain,drange)
plt.legend()
fig.savefig("figures/SI_figures/Sympy Spline interpolations.png")