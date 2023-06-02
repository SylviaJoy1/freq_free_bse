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


'''
Dynamical screening BSE, with or without TDA, singlet or triplet excitations. Density-fitted.
'''

from functools import reduce
import time
import tempfile
from functools import reduce
import numpy
import numpy as np
from scipy.optimize import newton

from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf import dft
from pyscf import gw
from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, _mo_without_core
from pyscf import __config__
#import line_profiler
from pyscf import symm
from pyscf.scf import hf_symm
from pyscf.data import nist

from pyscf import scf
import math

print(scf.__file__)

einsum = lib.einsum

REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)
# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)
MO_BASE = getattr(__config__, 'MO_BASE', 1)

#@profile
def kernel(bse, Omega_0, X_0, e_GW=None, td_e=None, td_xy=None, eris=None):
    '''dynamical screening BSE excitation energies

    Returns:
        A list :  converged, excitation energies
    '''
    # mf must be DFT; for HF use xc = 'hf'
    assert(bse.frozen == 0 or bse.frozen is None)

    cput0 = (time.clock(), time.time())
    log = logger.Logger(bse.stdout, bse.verbose)
    if bse.verbose >= logger.WARN:
        bse.check_sanity()
    bse.dump_flags()
    
    nocc = bse.nocc
    nmo = bse.nmo
    nvir = nmo - nocc
    
    if eris is None:
        eris = bse.ao2mo()
    if td_e is None:
        td_e = bse._tdscf.e
    if td_xy is None:
        td_xy = bse._tdscf.xy
    if e_GW is None:
        e_GW = bse.e_GW
    
    nroots = len(Omega_0)
    
    conv = []
    es = []
    vs = []
    #perturbative dynamical correction from Loos/Blase paper. bse.TDA=True only.
    if bse.perturbative:
        for Omega_0_S, X_0_S in zip(Omega_0, X_0):
            if not bse.TDA: 
                raise NotImplementedError
            #starting vector may be from non-TDA static screening BSE
            X_0_S = X_0_S[:nocc*nvir]
            de = 1e-6
            Omega_1_S = np.dot(X_0_S.T, A1(bse, Omega_0_S, X_0_S, eris, e_GW, td_e, td_xy))
            dK = np.dot(X_0_S.T, A1(bse, Omega_0_S+de, X_0_S, eris, e_GW, td_e, td_xy)).real - Omega_1_S.real
            Z_S = 1.0/(1.0-dK/de)
            es.append(Omega_0_S + Z_S*Omega_1_S.real)
            conv.append(True)
    #self-consistent dynamical. bse.TDA=True only for dynamical correction (so far)
    elif bse.sc:
        for Omega_0_S, X_0_S in zip(Omega_0, X_0):
            if bse.TDA:
                X_0_S = X_0_S[:nocc*nvir]
            else:
                raise NotImplementedError
            previous_e = 0
            Omega_S = Omega_0_S
            X_S = X_0_S
            it = 0
            while (it < 100) and (abs(previous_e - Omega_S) > 1e-6):
                previous_e = Omega_S
                convn, Omega_S, X_S = solve_A(bse, Omega_S, X_S, eris, e_GW, td_e, td_xy)
                it += 1
            conv.append(convn)
            es.append(Omega_S)
            vs.append(X_S)
    else:
        raise NotImplementedError

    if bse.verbose >= logger.INFO:
        np.set_printoptions(threshold=nocc*nvir)
        logger.debug(bse, '  BSE excitation energies =\n%s', es)
        for n, Omega_S, convn in zip(range(nroots), es, conv):
            logger.info(bse, 'BSE root %d E = %.16g  conv = %s', n, Omega_S*27.2114, convn)
        log.timer('BSE', *cput0) 
    return conv, es, vs
    
#@profile
def A1(bse, omega, r, eris=None, e_GW=None, td_e=None, td_xy=None):
    nmo = bse.nmo
    nocc = bse.nocc
    nvir = nmo - nocc
    
    if eris is None:
        eris = bse.ao2mo()
    if td_e is None:
        td_e = bse._tdscf.e
    if td_xy is None:
        td_xy = bse._tdscf.xy
    if e_GW is None:
        e_GW = bse.e_GW

    
    nexc = td.nstates
    #X_ia + Y_ia
    td_z = np.sum(np.asarray(td_xy), axis=1).reshape(nexc,nocc,nvir)*math.sqrt(2)
    tdm_oo = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Loo)
    tdm_vv = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Lvv)
    
    e_GW_occ = e_GW[:nocc]
    e_GW_vir = e_GW[nocc:]
        
    eta = 1e-3
    
    #W_stat - W_dynamical
    r1 = r[:nocc*nvir].copy().reshape(nocc,nvir)
    Hr1=np.zeros((nocc,nvir), dtype=complex)
    Hr1 -= 4*einsum('Zij,Zab,Z,jb->ia', tdm_oo, tdm_vv, 1./td_e, r1)
    #for A
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    Hr1[i,a] -= r1[j,b] * 2 * np.sum(tdm_vv[:,a,b]*tdm_oo[:,i,j]/(omega - td_e[:] -(e_GW_vir[b]-e_GW_occ[i]) + 1j*eta))
                    Hr1[i,a] -= r1[j,b] * 2 * np.sum(tdm_vv[:,a,b]*tdm_oo[:,i,j]/(omega - td_e[:] -(e_GW_vir[a]-e_GW_occ[j]) + 1j*eta))
    
    if bse.TDA:
        return Hr1.ravel()
    else:
        return NotImplementedError
        # tdm_ov = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Lov)
        # tdm_vo = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Lvo)
    
        # r2 = r[nocc*nvir:].copy().reshape(nocc,nvir)
    
        # #for B
        # Hr1 -= 4*einsum('Zib,Zaj,Z,jb->ia', tdm_ov, tdm_vo, 1./td_e, r2)
        # for i in range(nocc):
        #     for a in range(nvir):
        #         for j in range(nocc):
        #             for b in range(nvir):
        #                 Hr1[i,a] -= r2[j,b] * 2 * np.sum(tdm_vo[:,a,j]*tdm_ov[:,i,b]/(omega - td_e[:] -(e_GW_occ[j]-e_GW_occ[i]) + 1j*eta))
        #                 Hr1[i,a] -= r2[j,b] * 2 * np.sum(tdm_vo[:,a,j]*tdm_ov[:,i,b]/(omega - td_e[:] -(e_GW_vir[a]-e_GW_vir[b]) + 1j*eta))
        
        # #for -A
        # Hr2 = np.zeros((nocc,nvir), dtype=complex)
        # Hr2 -= 4*einsum('Zij,Zab,Z,jb->ia', tdm_oo, tdm_vv, 1./td_e, r2)
        # #for A
        # for i in range(nocc):
        #     for a in range(nvir):
        #         for j in range(nocc):
        #             for b in range(nvir):
        #                 Hr2[i,a] -= r2[j,b] * 2 * np.sum(tdm_vv[:,a,b]*tdm_oo[:,i,j]/(omega - td_e[:] -(e_GW_vir[b]-e_GW_occ[i]) + 1j*eta))
        #                 Hr2[i,a] -= r2[j,b] * 2 * np.sum(tdm_vv[:,a,b]*tdm_oo[:,i,j]/(omega - td_e[:] -(e_GW_vir[a]-e_GW_occ[j]) + 1j*eta)) 
        
        # #for -B
        # Hr2 -= 4*einsum('Zib,Zaj,Z,jb->ia', tdm_ov, tdm_vo, 1./td_e, r1)
        # for i in range(nocc):
        #     for a in range(nvir):
        #         for j in range(nocc):
        #             for b in range(nvir):
        #                 Hr2[i,a] -= r1[j,b] * 2 * np.sum(tdm_vo[:,a,j]*tdm_ov[:,i,b]/(omega - td_e[:] -(e_GW_occ[j]-e_GW_occ[i]) + 1j*eta))
        #                 Hr2[i,a] -= r1[j,b] * 2 * np.sum(tdm_vo[:,a,j]*tdm_ov[:,i,b]/(omega - td_e[:] -(e_GW_vir[a]-e_GW_vir[b]) + 1j*eta))
            
        # return np.hstack((Hr1.ravel(), -Hr2.ravel()))

from static_BSE_nstates import matvec as A0

def A_diag(bse, eris=None, e_GW=None, td_e=None, td_xy=None):
    nmo = bse.nmo
    nocc = bse.nocc
    nvir = nmo - nocc
    
    if eris is None:
        eris = bse.ao2mo()
    if td_e is None:
        td_e = bse._tdscf.e
    if td_xy is None:
        td_xy = bse._tdscf.xy
    if e_GW is None:
        e_GW = bse.e_GW

    nexc = td.nstates
    #X_ia + Y_ia
    td_z = np.sum(np.asarray(td_xy), axis=1).reshape(nexc,nocc,nvir)*math.sqrt(2)
    tdm_oo = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Loo)
    tdm_vv = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Lvv)
    
    e_GW_occ = e_GW[:nocc]
    e_GW_vir = e_GW[nocc:]
        
    eta = 1e-3
    
    diag=np.zeros((nocc,nvir))
    for i in range(nocc):
        for a in range(nvir):
            diag[i,a] += e_GW_vir[a] - e_GW_occ[i]
            diag[i,a] -= einsum('Q,Q', eris.Loo[:,i,i], eris.Lvv[:,a,a])
            if bse.singlet:
                diag[i,a] += 2*einsum('Q,Q', eris.Lov[:,i,a], eris.Lov[:,i,a])
            #Maybe use static version here
            #diag[i,a] -= 4 * np.sum(tdm_vv[:,a,a]*tdm_oo[:,i,i]/(Omega - td_e[:] -(e_GW_vir[a]-e_GW_occ[i]) + 1j*eta))
            diag[i,a] += 4 * np.sum(tdm_vv[:,a,a]*tdm_oo[:,i,i]/td_e[:])
            
    return diag.ravel()

def solve_A(bse, Omega, guess, eris=None, e_GW=None, td_e=None, td_xy=None):
    nmo = bse.nmo
    nocc = bse.nocc
    nvir = nmo - nocc
    
    if eris is None:
        eris = bse.ao2mo()
    if td_e is None:
        td_e = bse._tdscf.e
    if td_xy is None:
        td_xy = bse._tdscf.xy
    if e_GW is None:
        e_GW = bse.e_GW
   
    max_space = 100
    max_cycle = 200
    conv_tol = 1e-7
    
    # GHF or customized RHF/UHF may be of complex type
    real_system = (bse._scf.mo_coeff[0].dtype == np.double)
    
    diag = A_diag(bse, eris, e_GW, td_e, td_xy)
    matvec = lambda xs: [A0(bse, x, td_e, td_xy, e_GW, eris) + A1(bse, Omega, x, eris, e_GW, td_e, td_xy) for x in xs]
    
    def precond(r, e0, x0):
        return r/(e0-diag+1e-8)

    eig = lib.davidson_nosym1
    
    guess=[np.asarray(guess)]
    nroots = 1 
    def eig_close_to_init_guess(w, v, nroots, envs):
            x0 = lib.linalg_helper._gen_x0(envs['v'], envs['xs'])
            #print(type(max_memory))
            s = np.dot(np.asarray(guess).conj(), np.asarray(x0).T)
            snorm = np.einsum('pi,pi->i', s.conj(), s)
            idx = np.argsort(-snorm)[:nroots]
            return lib.linalg_helper._eigs_cmplx2real(w, v, idx, real_system)
    conv, e, v = eig(matvec, guess, precond, pick=eig_close_to_init_guess,\
                            tol=conv_tol, max_cycle=max_cycle,\
                            max_space=max_space, nroots=nroots)
    return conv, e[0], v[0]

def analyze(bse, verbose=None):
    if bse.perturbative: 
        raise NotImplementedError
    log = logger.new_logger(bse, verbose)
    mol = bse.mol
    mo_coeff = bse._scf.mo_coeff
    mo_occ = bse._scf.mo_occ
    nocc = np.count_nonzero(mo_occ == 2)
    nmo = bse.nmo
    nvir = nmo - nocc

    e_ev = np.asarray(bse.es) * nist.HARTREE2EV
    e_wn = np.asarray(bse.es) * nist.HARTREE2WAVENUMBER
    wave_length = 1e7/e_wn

    if bse.singlet:
        log.note('\n** Singlet excitation energies and oscillator strengths **')
    else:
        log.note('\n** Triplet excitation energies and oscillator strengths **')

    if mol.symmetry:
        orbsym = hf_symm.get_orbsym(mol, mo_coeff)
        from pyscf.symm import direct_prod
        x_sym = direct_prod(orbsym[mo_occ==2], orbsym[mo_occ==0], mol.groupname)
    else:
        x_sym = None


    for i, ei in enumerate(bse.vs):
        x = bse.vs[i][:nocc*nvir].reshape((nocc, nvir))
        if x_sym is None:
            log.note('Excited State %3d: %12.5f eV %9.2f nm',
                      i+1, e_ev[i], wave_length[i])
        else:
            wfnsym = analyze_wfnsym(bse, x_sym, x)
            log.note('Excited State %3d: %4s %12.5f eV %9.2f nm ',
                      i+1, wfnsym, e_ev[i], wave_length[i])

        if log.verbose >= logger.INFO:
            o_idx, v_idx = np.where(abs(x) > 0.1)
            for o, v in zip(o_idx, v_idx):
                log.info('    %4d -> %-4d %12.5f',
                          o+MO_BASE, v+MO_BASE+nocc, x[o,v])
    return bse

def analyze_wfnsym(bse, x_sym, x):
    '''Guess the wfn symmetry of BSE X amplitude'''
    if bse.perturbative: 
        raise NotImplementedError
    possible_sym = x_sym[(x > 1e-7) | (x < -1e-7)]
    if np.all(possible_sym == symm.MULTI_IRREPS):
        if bse.wfnsym is None:
            wfnsym = '???'
        else:
            wfnsym = bse.wfnsym
    else:
        ids = possible_sym[possible_sym != symm.MULTI_IRREPS]
        ids = np.unique(ids)
        if ids.size == 1:
            wfnsym = symm.irrep_id2name(bse.mol.groupname, ids[0])
        else:
            wfnsym = '???'
    return wfnsym

class BSE(lib.StreamObject):
    '''spatial orbital dynamical screening BSE

    Attributes:
        TDA : bool
            Whether to use the Tamm-Dancoff approximation to the BSE.  Default is True.
        singlet : bool
            Whether the excited state is a singlet or triplet.  Default is True.    
    Saved results:
        converged : bool
            Whether BSE roots are converged or not
        e : float
            BSE eigenvalues (excitation energies)  
    '''

    eta = getattr(__config__, 'bse_bse_BSE_eta', 1e-3)
    perturbative = getattr(__config__, 'bse_bse_BSE_perturbative', False)
    linearized = getattr(__config__, 'bse_bse_BSE_linearized', False)
    sc = getattr(__config__, 'bse_bse_BSE_sc', False)


    def __init__(self, mf, tdmf, e_GW, frozen=0, TDA=True, singlet=True):
        assert(isinstance(mf, dft.rks.RKS) or isinstance(mf, dft.rks_symm.SymAdaptedRKS)) #or isinstance(mf, scf.rhf.RHF) 
        self.mol = mf.mol
        self._scf = mf
        self.mf = mf
        self._tdscf = tdmf
        self.e_GW = e_GW
        #self.eta = 1e-3
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        #TODO: implement frozen orbs
        #self.frozen = frozen
        self.frozen = 0
        self.TDA = TDA
        self.singlet = singlet

##################################################
# don't modify the following attributes, they are not input options
        self.converged = False
        self.es = None
        self.vs = None
        self._nocc = None
        self._nmo = None
        self.mo_energy = None
        self.mo_coeff = mf.mo_coeff
        self.mo_occ = mf.mo_occ

        keys = set(('eta', 'perturbative', 'linearized', 'sc'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('method = %s', self.__class__.__name__)
        nocc = self.nocc
        nvir = self.nmo - nocc
        log.info('GW nocc = %d, nvir = %d', nocc, nvir)
        if self.frozen is not 0:
            log.info('frozen orbitals %s', str(self.frozen))
        logger.info(self, 'Loos/Blase perturbative dynamical = %s', self.perturbative)
        logger.info(self, 'self-consistent dynamical = %s', self.sc)
        logger.info(self, 'BSE within the TDA = %s', self.TDA)
        logger.info(self, 'singlet = %s', self.singlet)
        return self

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

    def kernel(self, Omega_0, X_0, eris=None, e_GW=None, td_e=None, td_xy=None):
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        
        if eris is None:
            eris = self.ao2mo()
        if td_e is None:
            td_e = self._tdscf.e
        if td_xy is None:
            td_xy = self._tdscf.xy
        if e_GW is None:
            e_GW = self.e_GW
        
        self.dump_flags()
        self.converged, self.es, self.vs= \
                kernel(self, Omega_0, X_0, e_GW, td_e, td_xy, eris)

        return self.converged, self.es, self.vs
    
    analyze = analyze
    analyze_wfnsym = analyze_wfnsym

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._scf.reset(mol)
        self._tdscf.reset(mol)
        return self

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        mem_incore = 2*(nocc**3*nvir+2*nocc**2*nvir**2+nocc*nvir**3) * 8/1e6
        mem_now = lib.current_memory()[0]
        self.mol.incore_anyway = True
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            raise NotImplementedError 


class _PhysicistsERIs:
    '''<pq|rs> not antisymmetrized
    
    This is gccsd _PhysicistsERIs without vvvv and without antisym'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None
        self.e_hf = None
        self.orbspin = None

        self.oooo = None
        self.ooov = None
        self.oovv = None
        self.ovvo = None
        self.ovov = None
        self.ovvv = None

    def _common_init_(self, mygw, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mygw.mo_coeff
        mo_idx = mygw.get_frozen_mask()
        # if getattr(mo_coeff, 'orbspin', None) is not None:
        #     self.orbspin = mo_coeff.orbspin[mo_idx]
        #     mo_coeff = lib.tag_array(mo_coeff[:,mo_idx], orbspin=self.orbspin)
        # else:
        #     orbspin = scf.ghf.guess_orbspin(mo_coeff)
        #     mo_coeff = mo_coeff[:,mo_idx]
        #     if not np.any(orbspin == -1):
        #         self.orbspin = orbspin[mo_idx]
        #         mo_coeff = lib.tag_array(mo_coeff, orbspin=self.orbspin)
        self.mo_coeff = mo_coeff

        self.mo_energy = mygw._scf.mo_energy
        dm = mygw._scf.make_rdm1(mygw.mo_coeff, mygw.mo_occ)
        vj, vk = mygw._scf.get_jk(mygw.mol, dm) 
        vj = reduce(np.dot, (mo_coeff.conj().T, vj, mo_coeff))
        self.vk = 0.5*reduce(np.dot, (mo_coeff.conj().T, vk, mo_coeff))
        vxc = mygw._scf.get_veff(mygw.mol, dm)
        self.vxc = reduce(np.dot, (mo_coeff.conj().T, vxc, mo_coeff)) - vj
        # Note: Recomputed fock matrix since SCF may not be fully converged.
        #dm = mygw._scf.make_rdm1(mygw.mo_coeff, mygw.mo_occ)
        #vhf = mygw._scf.get_veff(mygw.mol, dm)
        #fockao = mygw._scf.get_fock(vhf=vhf, dm=dm)
        #self.fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))
        #self.e_hf = mygw._scf.energy_tot(dm=dm, vhf=vhf)
        self.nocc = mygw.nocc
        self.mol = mygw.mol

        mo_e = self.mo_energy
        gap = abs(mo_e[:self.nocc,None] - mo_e[None,self.nocc:]).min()
        if gap < 1e-5:
            logger.warn(mygw, 'HOMO-LUMO gap %s too small for GCCSD', gap)
        return self
     
from pyscf import df
from pyscf.ao2mo import _ao2mo
#from RCCSD.
def _make_eris_incore(mygw, mo_coeff=None, ao2mofn=None):
    #cput0 = (time.clock(), time.time())
    eris = _PhysicistsERIs()
    eris._common_init_(mygw, mo_coeff)
    nocc = mygw.nocc
    nmo = mygw.nmo
    #nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    
    #from dfccsd. NOW IN CHEMIST NOTATION.
    def _init_df_eris():
        with_df = df.DF(mf.mol, auxbasis='def2-tzvp-ri' )
        naux = with_df.get_naoaux()        
        Loo = np.empty((naux,nocc,nocc))
        Lov = np.empty((naux,nocc,nvir))
        Lvv = np.empty((naux, nvir, nvir))
        mo = np.asarray(eris.mo_coeff, order='F')
        ijslice = (0, nmo, 0, nmo)
        p1 = 0
        Lpq = None
        for k, eri1 in enumerate(with_df.loop()):
            Lpq = _ao2mo.nr_e2(eri1, mo, ijslice, aosym='s2', mosym='s1', out=Lpq)
            p0, p1 = p1, p1 + Lpq.shape[0]
            Lpq = Lpq.reshape(p1-p0,nmo,nmo)
            Loo[p0:p1] = Lpq[:,:nocc,:nocc]
            Lov[p0:p1] = Lpq[:,:nocc,nocc:]
            Lvv[p0:p1] = Lpq[:, nocc:, nocc:]
        Lpq = None
        #Lvo = Lov.transpose(0,2,1).reshape(naux,nvir*nocc)
        Lov = Lov.reshape(naux,nocc*nvir)
        Loo = Loo.reshape(naux,nocc*nocc)
        Lvv = Lvv.reshape(naux,nvir*nvir)
        
        Lov = Lov.reshape(naux,nocc,nvir)
        #Lvo = Lvo.reshape(naux,nvir,nocc)
        Loo = Loo.reshape(naux,nocc,nocc)
        Lvv = Lvv.reshape(naux,nvir,nvir)
        return Lov, Loo, Lvv
    eris.Lov, eris.Loo, eris.Lvv = _init_df_eris()
    
    return eris

if __name__ == '__main__':
    from pyscf import dft, tddft
    from pyscf import gto
    import static_BSE_nstates as static_BSE
    from pyscf import scf
    
    # mol = gto.Mole(units = 'A')
    # mol.verbose = 5
    # mol.atom = [['N',(0.0000, 0.0000, 0.0000)],
    #         ['N', (0.0000, 0.0000, 1.100779955618)]]
    # mol.basis = 'cc-pVDZ'
    # mol.cart = True
    # mol.symmetry = True
    # mol.build()
    
    # mf = dft.RKS(mol)
    # mf.xc = 'hf'
    # mf.kernel()

    # nocc = mol.nelectron//2
    # nmo = mf.mo_energy.size
    # nvir = nmo-nocc

    # td = tddft.dRPA(mf)
    # td.nstates = nocc*nvir
    # td.kernel()
    
    # mygw = gw.GW(mf, td)
    # mygw.linearized = True
    # mygw.kernel(orbs=range(nmo))
    # e_GW = mygw.mo_energy
    
    
    # # Reference values: Table 1 of https://doi.org/10.1063/5.0023168
    
    # ##########################
    # singlet = True
    
    # #static part
    # static_bse = static_BSE.BSE(mf, td, e_GW, TDA=False, singlet=singlet)
    
    # converged, excitations, Omega_0, X_0  = static_bse.kernel(nstates = 2)
    # assert all(converged)
    
    # bse = BSE(mf, td, e_GW, TDA=True, singlet=singlet)
    # bse.perturbative = True
    # #perturbative dynamical correction. bse.TDA=True only.
    # conv, exc_energy = bse.kernel(Omega_0, X_0)
    #exc_energy[0]: 9.371453707603012 
    #exc_energy[1]: 10.05497104937913
    
    ############################
    # singlet = False
    
    # #static part
    # static_bse = static_BSE.BSE(mf, td, e_GW, TDA=False, singlet=singlet)
    
    # converged, excitations, Omega_0, X_0  = static_bse.kernel(nstates = 2)
    # assert all(converged)
    
    # bse = BSE(mf, td, e_GW, TDA=True, singlet=singlet)
    # bse.perturbative = True
    # #perturbative dynamical correction. bse.TDA=True only.
    # conv, exc_energy = bse.kernel(Omega_0, X_0)
    # #exc_energy[0]: 6.934981533532951 
    # #exc_energy[1]: 8.156862684054602 
    
    ###########################
    # singlet = True
    
    # #static part
    # static_bse = static_BSE.BSE(mf, td, e_GW, TDA=False, singlet=singlet)
    
    # converged, excitations, Omega_0, X_0  = static_bse.kernel(nstates = 10)
    # assert all(converged)
    
    # bse = BSE(mf, td, e_GW, TDA=True, singlet=singlet)
    # bse.perturbative = True
    # #perturbative dynamical correction. bse.TDA=True only.
    # conv, exc_energy = bse.kernel(Omega_0, X_0)
    
    #convergence problems are probably due to small max_space and/or max_cycle.
    
    # ###########################
    # singlet = True
    
    # #static part
    # static_bse = static_BSE.BSE(mf, td, e_GW, TDA=False, singlet=singlet)
    
    # converged, excitations, Omega_0, X_0  = static_bse.kernel(nstates = 2)
    # assert all(converged)
    # static_bse.analyze()
    
    # bse = BSE(mf, td, e_GW, TDA=True, singlet=singlet)
    # bse.sc = True
    # #self-consistent dynamical correction. bse.TDA=True only for now.
    # conv, es, vs = bse.kernel(Omega_0, X_0)
    # # exc_energy[0]: 9.392640299465404
    # # exc_energy[1]: 10.05291481019241
    # bse.analyze()
    
    ###########################
    # singlet = True
    
    # #static part
    # static_bse = static_BSE.BSE(mf, td, e_GW, TDA=False, singlet=singlet)
    
    # converged, excitations, Omega_0, X_0  = static_bse.kernel(nstates = 10)
    # assert all(converged)
    
    # bse = BSE(mf, td, e_GW, TDA=True, singlet=singlet)
    # bse.sc = True
    # #self-consistent dynamical correction. bse.TDA=True only for now.
    # conv, exc_energy = bse.kernel(Omega_0, X_0)
    
    ##########################
    ##########################
    
    mol = gto.Mole(units = 'A')
    mol.verbose = 5
#formaldehyde from Thiel set 
#     mol.atom = '''H      0.000000    0.934473    -0.588078
# H      0.000000    -0.934473    -0.588078
# C      0.000000    0.000000     0.000000
# O      0.000000    0.000000     1.221104;'''
#cyclopropene from Thiel set 
    mol.atom = '''
  H           0.912650        0.000000         1.457504
  H          -0.912650        0.000000         1.457504
  H           0.000000       -1.585659       -1.038624
  H           0.000000        1.585659       -1.038624
  C           0.000000        0.000000         0.859492
  C           0.000000       -0.651229       -0.499559
  C           0.000000        0.651229       -0.499559;'''
    mol.basis = 'def2-TZVP'
    mol.cart = True
    mol.symmetry = True
    mol.build()
    
    mf = dft.RKS(mol)
    mf.xc = 'hf'
    # mf.xc = 'pbe0'
    mf.kernel()

    nocc = mol.nelectron//2
    nmo = mf.mo_energy.size
    nvir = nmo-nocc

    td = tddft.dRPA(mf)
    td.nstates = nocc*nvir
    td.kernel()

    mygw = gw.GW(mf, td)
    mygw.linearized = True
    mygw.kernel(orbs=range(nmo))
    e_GW = mygw.mo_energy
    
    
    #Reference values: 
    
    ###########################
    singlet = True
    
    static_bse = static_BSE.BSE(mf, td, e_GW, TDA=False, singlet=singlet)
    converged, excitations, Omega_0_s, X_0_s  = static_bse.kernel(nstates = 6)
    assert all(converged)
    static_bse.analyze()
    
    bse = BSE(mf, td, e_GW, TDA=True, singlet=singlet)
    # bse.sc = True
    bse.perturbative = True
    #perturbative dynamical correction. bse.TDA=True only.
    conv, es, vs = bse.kernel(Omega_0_s, X_0_s)
    # bse.analyze()