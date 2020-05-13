import matplotlib.pyplot as plt
import protease_activity_analysis as paa

fig1, ax1 = plt.subplots(1)
ax1.plot([3,1,4,1,5], 'ks-', mec='w', mew=5, ms=20)

fig2, ax2 = plt.subplots(1)
ax2.plot([1,1,2], 'gs-', mec='w', mew=5, ms=20)

figs = [fig1, fig2]
paa.vis.render_html_figures(figs)
