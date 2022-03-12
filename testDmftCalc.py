from dmft_calc import *

beta = 10
t = 1
U = 4
Nk = 400
mu = U/2
ks = np.linspace(-np.pi,np.pi,Nk,endpoint = False)
gf_struct = [('up', [0]), ('down', [0])]
epsk = lambda k:-2*t*np.cos(k)
dispersions = {('up',0) : epsk, ('down',0) : epsk}
h_int = U*tops.n('up',0)*tops.n('down',0)
nloops = 2
C10 = DmftCalc(beta,gf_struct,dispersions,h_int,mu,ks,defaultSolverInputs = {'n_cycles' : 10000, 'length_cycle' : 100, 'n_warmup_cycles' : 2000})
C10.runDmftLoop(nloops = nloops)
C20 = DmftCalc(beta,gf_struct,dispersions,h_int,mu,ks,defaultSolverInputs = {'n_cycles' : 20000, 'length_cycle' : 100, 'n_warmup_cycles' : 4000})
C20.runDmftLoop(nloops = nloops)
C = DmftCalc(beta,gf_struct,dispersions,h_int,mu,ks,defaultSolverInputs = {'n_cycles' : 10*40000, 'length_cycle' : 100, 'n_warmup_cycles' : 10*8000})
C.runDmftLoop(nloops = 10)
oplot(C.S.G_iw['up'],label = "hubbard")
#C80 = DmftCalc(beta,gf_struct,dispersions,h_int,mu,ks,defaultSolverInputs = {'n_cycles' : 80000, 'length_cycle' : 100, 'n_warmup_cycles' : 16000})
#C80.runDmftLoop(nloops = 20)
#C160 = DmftCalc(beta,gf_struct,dispersions,h_int,mu,ks,defaultSolverInputs = {'n_cycles' : 160000, 'length_cycle' : 100, 'n_warmup_cycles' : 32000})
#C160.runDmftLoop(nloops = 20)

"""plt.plot(C10.mudensities[mu],label = "10k cycles")
plt.plot(C20.mudensities[mu],label = "20k cycles")
plt.plot(C.mudensities[mu],label = "40k cycles")
#plt.plot(C80.mudensities[mu],label = "80k cycles")
#plt.plot(C160.mudensities[mu],label = "160k cycles")
plt.xlabel("dmft loop iteration")
plt.ylabel("calculated density")
plt.title("Density Convergence (U = 4, $\mu$ = U/2)")
plt.legend()
plt.show()"""

# example with Sigma eph
omega = 0.2
g = 0.1
Sigma_ephk = {k:C.S.Sigma_iw.copy() for k in ks}

k0 = ks[0]
Sigma_ephk[k0].zero()
for q in ks:
    kpq = ((k0 + q) + np.pi) % (2*np.pi) - np.pi
    epskq = dispersions[('up',0)](kpq)
    Nq = 1 / (np.exp(omega*beta) - 1)
    fkpq = 1 / (np.exp(epskq*beta) + 1)
    Sigma_ephk[k0]['up'][0,0] << Sigma_ephk[k0]['up'][0,0] + (g**2)*((Nq + fkpq)*inverse(iOmega_n - epskq + omega) - (Nq + 1 - fkpq)*inverse(iOmega_n - epskq - omega))
    #Sigma_ephk[k]['down'][0,0] << Sigma_ephk[k]['down'][0,0] + (g**2)*((Nq + fkpq)*inverse(iOmega_n - epskq + omega) - (Nq + 1 - fkpq)*inverse(iOmega_n - epskq - omega))
Sigma_ephk[k0]['down'][0,0] << Sigma_ephk[k0]['up'][0,0]

for k in ks[1:]:
    Sigma_ephk[k].zero()
    Sigma_ephk[k] << Sigma_ephk[k0]
    # works since they are k independent

for mufact in [1.0,1.2,1.4,1.5,1.6]:
    C.updatemu(U/2 * mufact,Sigmak = lambda k : Sigma_ephk[k])
    C.runDmftLoop(nloops = 10, Sigmak = lambda k: Sigma_ephk[k])
    oplot(C.S.G_iw['up'],label = "hubbard holstein")
