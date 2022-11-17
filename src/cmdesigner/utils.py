import numpy as np
import numpy.polynomial.polynomial as P
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def splot (polynom, * , x_lims, save_path = None ):
    _, F_s = polynom.s11_num
    _, P_s = polynom.s21_num
    _, E_s = polynom.denum
    e_r, e = polynom.ripple_factors
    complex_frequency = x_lims * 1j

    S11 = 20 * np.log10(abs(P.polyval(complex_frequency, F_s) /
                        (P.polyval(complex_frequency, E_s) * e_r)))
    S21 = 20 * np.log10(abs(P.polyval(complex_frequency, P_s) /
                        (P.polyval(complex_frequency, E_s) * e)))

    _, ax = plt.subplots()
    ax.plot(x_lims, S11, c='y', label='S11')
    ax.plot(x_lims, S21, label='S21')
    ax.set_ylim([-50, 1])
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(r'|S11| and |S21| (dB)')
    ax.grid(linestyle='--')
    ax.legend()

    ax2 = plt.axes([0, 0, 1, 1])
    ip = InsetPosition(ax, [0.72, 0.6, 0.25, 0.25])
    ax2.set_axes_locator(ip)
    ax2.set_xlim([0.3, 1])
    ax2.set_ylim([-0.05, 0.01])
    ax2.plot(x_lims, S21, label='S11')
    ax2.set_xticks([])
    mark_inset(ax, ax2, loc1=1, loc2=3, fc='none', ec='0.5')
    if save_path is None:
        plt.savefig('plot.png')