import numpy as np
from math import pi
from triqs.gf import *
import triqs.gf.tools as tools
from triqs.plot.mpl_interface import oplot,plt
from triqs import operators as tops
import triqs_cthyb
from h5 import *
import triqs.utility.mpi as mpi
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class DmftCalc():
    def __init__(self,beta, gf_struct, dispersions, h_int,mu, ks,solver = triqs_cthyb.Solver,defaultSolverInputs = {'n_cycles' : 20000, 'length_cycle' : 100, 'n_warmup_cycles' : 4000},verbosity = "high"):
        self.beta = beta
        self.gf_struct = gf_struct
        self.dispersions = dispersions
        self.h_int = h_int
        self.mu = mu
        self.mudensities = {mu:[]}
        self.ks = ks
        self.Nk = len(self.ks)
        self.S = solver(beta = self.beta, gf_struct = self.gf_struct)
        self.S.Sigma_iw << mu
        self.Gks = {k : self.S.G0_iw.copy() for k in ks}
        # ^ arguably, this should not be stored since it can be calculated from much less information in some cases
        self.timeSC,self.timeImp = 0.0,0.0
        self.defaultSolverInputs = defaultSolverInputs
        self.verbosity = verbosity
        self.updateGFs()

    def calcGk_iw(self, Sigmak = lambda k : 0,parallel = True):
        if parallel:
            ksprocs = {iproc : self.ks[iproc::size] for iproc in range(size)}
            for k in ksprocs[rank]:
                sigmak = self.S.Sigma_iw.copy()
                sigmak << sigmak + Sigmak(k)
                self.Gks[k].zero()
                for name,indices in self.gf_struct:
                    for i in indices:
                        self.Gks[k][name][i,i] << inverse(iOmega_n - self.dispersions[(name,i)](k) + self.mu - sigmak[name][i,i])
            for iproc in range(size):
                for k in ksprocs[iproc]:
                    self.Gks[k] << comm.bcast(self.Gks[k],root = iproc)
        else:
            for k in self.ks:
                sigmak = self.S.Sigma_iw.copy()
                sigmak << sigmak + Sigmak(k)
                self.Gks[k].zero()
                for name,indices in self.gf_struct:
                    for i in indices:
                        self.Gks[k][name][i,i] << inverse(iOmega_n - self.dispersions[(name,i)](k) + self.mu - sigmak[name][i,i])

    def calcGimp_iw(self):
        self.S.G0_iw.zero()
        Gimp = self.S.G0_iw.copy()
        for k in self.ks:
            Gimp << Gimp + self.Gks[k]
        Gimp << Gimp / self.Nk
        self.S.G0_iw << tools.dyson(G_iw = Gimp, Sigma_iw = self.S.Sigma_iw)

    def updateGFs(self,Sigmak = lambda k : 0):
        self.calcGk_iw(Sigmak = Sigmak)
        self.calcGimp_iw()

    def solveImpurity(self,**kwargs):
        if kwargs == {}:
            kwargs = self.defaultSolverInputs
        self.S.solve(h_int = self.h_int,**kwargs)

    def runDmftLoop(self,nloops = 1,Sigmak = lambda k : 0,**solverkwargs):
        # ToDo: add ramp up capability for ncycles
        for i in range(nloops):
            tstart = time.time()
            self.solveImpurity(**solverkwargs)
            self.timeImp += time.time() - tstart
            tstart = time.time()
            self.updateGFs(Sigmak = Sigmak)
            self.timeSC += time.time() - tstart
            self.mudensities[self.mu].append(self.totalDensity())

    def totalDensity(self):
        density = self.S.G_iw.density()
        return sum([np.trace(density[name]).real for name, indices in self.gf_struct])

    def updatemu(self,mu,Sigmak = lambda k : 0):
        # resets solver with default self energy (= mu) and corresponding G0_imp
        self.mu = mu
        self.S.Sigma_iw << self.mu
        self.updateGFs(Sigmak = Sigmak)
        self.mudensities[self.mu] = []
