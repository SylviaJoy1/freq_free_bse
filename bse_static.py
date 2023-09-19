#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Timothy Berkelbach <tim.berkelbach@gmail.com>
#          Sylvia Bintrim <sjb2225@columbia.edu>
#

'''
static screening BSE with iterative diagonalization, with or without TDA, singlet or triplet excitations. Density-fitted.
'''

import time
from functools import reduce
import numpy as np
from pyscf import lib, scf, dft, tddft, gw, df, symm
from pyscf.lib import logger
from pyscf import __config__
import math
from pyscf.scf import hf_symm
from pyscf.data import nist

einsum = lib.einsum

from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
    
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)
# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)
MO_BASE = getattr(__config__, 'MO_BASE', 1)

def kernel(bse, eris, td_e, td_xy, gw_e, nstates=None, verbose=logger.NOTE):
    '''static screening BSE excitation energies

    Returns:
        A list :  converged, number of states, excitation energies, eigenvectors
    '''
    #mf must be DFT; for HF use xc = 'hf'
    assert(isinstance(bse.mf, dft.rks.RKS) or isinstance(bse.mf, dft.rks_symm.SymAdaptedRKS))
    assert(bse.frozen == 0 or bse.frozen is None)
    
    cput0 = (time.clock(), time.time())
    log = logger.Logger(bse.stdout, bse.verbose)
    if bse.verbose >= logger.WARN:
        bse.check_sanity()
    bse.dump_flags()

    nocc = bse.nocc
    nmo = bse.nmo
    nvir = nmo - nocc
    
    if nstates is None: nstates = 1
    
    matvec, diag = bse.gen_matvec(eris, td_e, td_xy, gw_e)
    
    size = nocc*nvir
    if not bse.TDA:
        size *= 2
    
    guess, nstates = bse.get_init_guess(gw_e, nstates)
        
    nroots = nstates

    def precond(r, e0, x0):
        return r/(e0-diag+1e-12)

    if bse.TDA:
        eig = lib.davidson1
    else: 
        eig = lib.davidson_nosym1
    
    # GHF or customized RHF/UHF may be of complex type
    real_system = (bse._scf.mo_coeff[0].dtype == np.double)
    
    def pickeig(w, v, nroots, envs):
        #real_idx = np.where(abs(w.imag) < 1e-3)[0]
        real_idx = np.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                              (w.real > POSTIVE_EIG_THRESHOLD))[0]
        return lib.linalg_helper._eigs_cmplx2real(w, v, real_idx, real_system)
    conv, e, xy = eig(matvec, guess, precond, pick=pickeig,
                       tol=bse.conv_tol, max_cycle=bse.max_cycle,
                       max_space=bse.max_space, nroots=nroots, verbose=log)
    xy =   [(xi[:nocc*nvir].reshape(nocc, nvir)*np.sqrt(.5), 0) for xi in xy]
        
    if bse.verbose >= logger.INFO:
        np.set_printoptions(threshold=nocc*nvir)
        logger.debug(bse, '  BSE excitation energies =\n%s', e.real)
        for n, en, vn, convn in zip(range(nroots), e, xy, conv):
            logger.info(bse, '  BSE root %d E = %.16g eV  conv = %s',
                        n, en*27.2114, convn)
        log.timer('BSE', *cput0) 
    
    return conv, nstates, e, xy
    
def matvec(bse, r, eris, td_e=None, td_xy=None, gw_e=None):
    '''matrix-vector multiplication'''
   
    nocc = bse.nocc
    nmo = bse.nmo
    nvir = nmo - nocc

    if td_e is None:
        td_e = bse._tdscf.e
    if td_xy is None:
        td_xy = bse._tdscf.xy
    if gw_e is None:
        gw_e = bse.gw_e

    nexc = nocc*nvir
    #X_ia + Y_ia
    td_z = np.sum(np.asarray(td_xy), axis=1).reshape(nexc,nocc,nvir)*math.sqrt(2)
    tdm_oo = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Loo)
    tdm_vv = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Lvv)
    tdm_ov = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Lov)
    
    gw_e_occ = gw_e[:nocc]
    gw_e_vir = gw_e[nocc:]
    
    r1 = r[:nocc*nvir].copy().reshape(nocc,nvir)
    #for A
    Hr1 = einsum('a,ia->ia', gw_e_vir, r1) - einsum('i,ia->ia', gw_e_occ, r1)
    Hr1 += 4*einsum('Zij,Zab,Z,jb->ia', tdm_oo, tdm_vv, 1./np.array(td_e), r1)
    Hr1 -= einsum('Qij, Qab,jb->ia', eris.Loo, eris.Lvv, r1)
    if bse.singlet:
        Hr1 += 2*einsum('Qia, Qjb,jb->ia', eris.Lov, eris.Lov, r1)
    
    if bse.TDA:
        return Hr1.ravel()
    
    else:
        r2 = r[nocc*nvir:].copy().reshape(nocc,nvir)
    
        #for B
        Hr1 += 4*einsum('Zib,Zja,Z,jb->ia', tdm_ov, tdm_ov, 1./td_e, r2)
        Hr1 -= einsum('Qib, Qja,jb->ia', eris.Lov, eris.Lov, r2)
        if bse.singlet:
            Hr1 += 2*einsum('Qia, Qjb,jb->ia', eris.Lov, eris.Lov, r2)
        
        #for -A
        Hr2 = einsum('a,ia->ia', gw_e_vir, r2) - einsum('i,ia->ia', gw_e_occ, r2)
        Hr2 += 4*einsum('Zij,Zab,Z,jb->ia', tdm_oo, tdm_vv, 1./td_e, r2)
        Hr2 -= einsum('Qij, Qab,jb->ia', eris.Loo, eris.Lvv, r2)
        if bse.singlet:
            Hr2 += 2*einsum('Qia, Qjb,jb->ia', eris.Lov, eris.Lov, r2) 
        
        #for -B
        Hr2 += 4*einsum('Zib,Zja,Z,jb->ia', tdm_ov, tdm_ov, 1./td_e, r1)
        Hr2 -= einsum('Qib, Qja,jb->ia', eris.Lov, eris.Lov, r1)
        if bse.singlet:
            Hr2 += 2*einsum('Qia, Qjb,jb->ia', eris.Lov, eris.Lov, r1)
            
        return np.hstack((Hr1.ravel(), -Hr2.ravel()))

from types import SimpleNamespace
from pyscf.ao2mo import _ao2mo
def get_Lpq(mf, df_file):
    mo_coeff = mf.mo_coeff
    nocc = mol.nelectron//2
    nmo = mf.mo_energy.size
    nvir = nmo - nocc

    with_df = df.DF(mol)
    mf.with_df._cderi = df_file

    naux = with_df.get_naoaux()

    Loo = np.empty((naux,nocc,nocc))
    Lov = np.empty((naux,nocc,nvir))
    Lvv = np.empty((naux,nvir,nvir))
    mo = np.asarray(mo_coeff, order='F')
    ijslice = (0, nmo, 0, nmo)
    p1 = 0
    Lpq = None
    for k, eri1 in enumerate(with_df.loop()):
        Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
        p0, p1 = p1, p1 + Lpq.shape[0]
        Lpq = Lpq.reshape(p1-p0,nmo,nmo)
        Loo[p0:p1] = Lpq[:,:nocc,:nocc]
        Lov[p0:p1] = Lpq[:,:nocc,nocc:]
        Lvv[p0:p1] = Lpq[:,nocc:,nocc:]

    eris = SimpleNamespace()
    eris.Loo = Loo
    eris.Lov = Lov
    eris.Lvv = Lvv
    
    return eris

from pyscf.tdscf import rhf
class BSE(rhf.TDA):
    '''static screening BSE

    Attributes:
        TDA : bool
            Whether to use the Tamm-Dancoff approximation to the BSE.  Default is True.
        singlet : bool
            Whether the excited state is a singlet or triplet.  Default is True.    
    Saved results:
        converged : bool
        nstates : int
        es : list
            BSE eigenvalues (excitation energies)
        vs : list
            BSE eigenvectors
        
    '''
     
    def __init__(self, mf, tdmf, gw_e, df_file=None, frozen=None, TDA=True, singlet=True, mo_coeff=None, mo_occ=None):
        assert(isinstance(mf, dft.rks.RKS) or isinstance(mf, dft.rks_symm.SymAdaptedRKS)) #or isinstance(mf, scf.rhf.RHF) 
        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ
        
        self.mf = mf
        self.mol = mf.mol
        self._scf = mf
        if df_file is None:
            self.mf.with_df = df.DF(mol)
            self.mf.with_df._cderi_to_save = 'DF_ints.h5'
            df_file = 'DF_ints.h5'
        self.df_file = df_file
        self._tdscf = tdmf
        self.gw_e = gw_e
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        self.max_space = getattr(__config__, 'eom_rccsd_EOM_max_space', 100)
        self.max_cycle = getattr(__config__, 'eom_rccsd_EOM_max_cycle', 200)
        self.conv_tol = getattr(__config__, 'eom_rccsd_EOM_conv_tol', 1e-7)

        self.frozen = frozen
        self.TDA = TDA
        self.singlet = singlet
##################################################
# don't modify the following attributes, they are not input options
        self.conv = False
        self.e = None
        self.xy = None
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self.eris = None
        self.nstates = None
        self._nmo = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not None:
            log.info('frozen = %s', self.frozen)
        logger.info(self, 'max_space = %d', self.max_space)
        logger.info(self, 'max_cycle = %d', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        logger.info(self, 'BSE within the TDA = %s', self.TDA)
        if getattr(self, 'TDA') is False:
            logger.warn(self, 'non-TDA BSE may not always converge (triplet instability problem).')
        logger.info(self, 'singlet = %s', self.singlet)
        return self
    
    def kernel(self, eris=None, td_e=None, td_xy=None, gw_e=None, nstates=None):
        if eris is None:
            eris = self.get_Lpq()
        if td_e is None:
            td_e = self._tdscf.e
        if td_xy is None:
            td_xy = self._tdscf.xy
        if gw_e is None:
            gw_e = self.gw_e
        self.conv, self.nstates, self.e, self.xy = kernel(self, eris, td_e, td_xy, gw_e, nstates)
        return self.conv, self.nstates, self.e, self.xy
    
    def get_Lpq(self):
        eris = get_Lpq(self.mf, self.df_file)
        self.eris = eris
        return eris
        
    matvec = matvec
    
    def gen_matvec(self, eris, td_e=None, td_xy=None, gw_e=None):
        if td_e is None:
            td_e = self._tdscf.e
        if td_xy is None:
            td_xy = self._tdscf.xy
        if gw_e is None:
            gw_e = self.gw_e
        diag = self.get_diag(eris, td_e, td_xy, gw_e)
        matvec = lambda xs: [self.matvec(x, eris, td_e, td_xy, gw_e) for x in xs]
        return matvec, diag

    def get_diag(self, eris, td_e=None, td_xy=None, gw_e=None):
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        
        if td_e is None:
            td_e = self._tdscf.e
        if td_xy is None:
            td_xy = self._tdscf.xy
        if gw_e is None:
            gw_e = self.gw_e
        
        gw_e_occ = gw_e[:nocc]
        gw_e_vir = gw_e[nocc:]
        
        nexc = self._tdscf.nstates
        #X_ia + Y_ia
        td_z = np.sum(np.asarray(td_xy), axis=1).reshape(nexc,nocc,nvir)*math.sqrt(2)
        tdm_oo = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Loo)
        tdm_vv = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Lvv)
        
        diag = np.zeros((nocc,nvir))
        for i in range(nocc):
            for a in range(nvir):
                diag[i,a] += gw_e_vir[a] - gw_e_occ[i]
                diag[i,a] += 4*np.sum(tdm_oo[:,i,i]*tdm_vv[:,a,a]/td_e[:])
                diag[i,a] -= np.dot(eris.Loo[:,i,i], eris.Lvv[:,a,a])
                if self.singlet:
                    diag[i,a] += 2*np.dot(eris.Lov[:,i,a], eris.Lov[:,i,a])
        diag = diag.ravel()
        if self.TDA:
            return diag
        else: 
            return np.hstack((diag, -diag))
        
    def get_init_guess(self, gw_e=None, nstates=None):
        
        if nstates is None: nstates = 1
        if gw_e is None: gw_e = self.gw_e
        
        Ediff = gw_e[None,self.nocc:] - gw_e[:self.nocc, None]
        e_ia = np.hstack([x.ravel() for x in Ediff])
        e_ia_max = e_ia.max()
        nov = e_ia.size
        nstates = min(nstates, nov)
        e_threshold = min(e_ia_max, e_ia[np.argsort(e_ia)[nstates-1]])
        # Handle degeneracy, include all degenerated states in initial guess
        e_threshold += 1e-6

        idx = np.where(e_ia <= e_threshold)[0]
        x0 = np.zeros((idx.size, nov))
        for i, j in enumerate(idx):
            x0[i, j] = 1  # Koopmans' excitations
        guess = x0
        if not self.TDA:
            guess = np.hstack((guess,0*guess))
        
        return guess, nstates

    @property
    def nocc(self):
        return self.get_nocc()
    @nocc.setter
    def nocc(self, n):
        self._nocc = n

    @property
    def nmo(self):
        return self.get_nmo()
    @nmo.setter
    def nmo(self, n):
        self._nmo = n

    get_nocc = get_nocc
    get_nmo = get_nmo
    get_frozen_mask = get_frozen_mask

if __name__ == '__main__':
    from pyscf import gto
    #doi: 10.1063/5.0023168
    mol = gto.Mole(unit='A')
    mol.atom = [['O',(0.0000, 0.0000, 0.0000)],
            ['H', (0.7571, 0.0000, 0.5861)],
            ['H', (-0.7571, 0.0000, 0.5861)]]
    mol.basis = 'aug-cc-pVTZ'
    mol.symmetry = True
    mol.build()
    formula = 'water'
    
    mf = dft.RKS(mol)
    mf.xc = 'hf'
    mf.kernel()
    
    nocc = mol.nelectron//2
    nmo = mf.mo_energy.size
    nvir = nmo-nocc
    
    td = tddft.dRPA(mf)
    td.nstates = nocc*nvir
    td.kernel()
    
    mygw = gw.GW(mf, freq_int='exact', tdmf=td)
    mygw.kernel(orbs=range(nmo))
    gw_e = mygw.mo_energy
    
    mf.with_df = df.DF(mol)
    mf.with_df._cderi_to_save = formula+'.h5'
    
    #cannot request too many states with TDA=False
    bse = BSE(mf, td, gw_e, df_file=formula+'.h5', TDA=False, singlet=True)
    conv, excitations, e, xy = bse.kernel(nstates=3)
    assert((27.2114*bse.e[0] - 8.09129 < 1e-3))
    assert((27.2114*bse.e[1] - 9.78553 < 1e-3))
    assert((27.2114*bse.e[2] - 10.41702  < 1e-3))
    bse.analyze()

    bse.singlet=False
    conv, excitations, e, xy = bse.kernel(nstates=3)
    assert((27.2114*bse.e[0] - 7.61802 < 1e-3))
    assert((27.2114*bse.e[0] - 9.59825 < 1e-3))
    assert((27.2114*bse.e[0] - 9.79518 < 1e-3))
    bse.analyze()
    
