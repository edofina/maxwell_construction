import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci
import scipy.optimize as sco
import scipy.signal as scs

# van der waals normalized function
def vdw(Tr, Vr):
    pV = 8*Tr/(3*Vr-1) - 3/Vr**2
    return pV
#applying maxwell construction
def vdwM(Tr, Vr):
    pV = vdw(Tr, Vr)
    if Tr >= 1:      #Tr values of < 1 yields unphysical values
        return pV
    #for the Van der Waal loop
    npmin = scs.argrelextrema(pV, np.less) #relative minimum for normalized pressure
    npmax = scs.argrelextrema(pV, np.greater) #relative maximum for normalized pressure
    Vr0 = np.mean([Vr[npmin], Vr[npmax]])
    
    #obtain min and max reduced volumes at pr0
    def vlims(pr0):
        eos = np.poly1d( (3*pr0, -(pr0+8*Tr), 9, -3) )
        roots = eos.r
        roots.sort()
        Vrmin, _, Vrmax = roots
        return Vrmin, Vrmax

    def area_diff(Vr0):
        pr0 = vdw(Tr, Vr0)
        Vrmin, Vrmax = vlims(pr0)
        return sci.quad(lambda vr: vdw(Tr, vr) - pr0, Vrmin, Vrmax)[0]

    Vr0 = sco.newton(area_diff, Vr0)
    pr0 = vdw(Tr, Vr0)
    Vrmin, Vrmax = vlims(pr0)

    #constant pressure in Van der Waal loop, pr0.
    pV[(Vr >= Vrmin) & (Vr <= Vrmax)] = pr0
    return pV

Vr = np.linspace(0.5, 3, 500)
Tc = 304 #critical temperature
def plot_pV(T):
    Tr = T / Tc
    ax.plot(Vr, vdw(Tr, Vr), lw=2, alpha=0.3)
    ax.plot(Vr, vdwM(Tr, Vr), lw=2, label='{:.2f}'.format(Tr))

fig, ax = plt.subplots()
for T in range(270, 320, 10):
    plot_pV(T)
    
plt.ylim(0,2)
plt.xlim(0.4,3)
plt.ylabel('$p$')
plt.xlabel('$V$')
plt.legend(title='Normalized temperature')
plt.show()
