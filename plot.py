import matplotlib.pyplot as plt
import parameters as param

plt.plot([1,2,3],[1,2,3],'ro')
plt.savefig("{0}/test.png".format(param.wsl_edrive), dpi=400)
