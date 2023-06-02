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
static screening BSE with iterative diagonalization, with or without TDA, singlet or triplet excitations. Density-fitted.
'''

import time
from functools import reduce
import numpy as np
from pyscf import lib
from pyscf import gto
#from pyscf import ao2mo
from pyscf import tddft
from pyscf.lib import logger
from pyscf import __config__
from pyscf import dft
from pyscf import gw
import math
from pyscf import scf
from pyscf import symm
from pyscf.scf import hf_symm
from pyscf.data import nist

einsum = lib.einsum

from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask
    
REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshoasld', 1e-4)
# Low excitation filter to avoid numerical instability
POSTIVE_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)
MO_BASE = getattr(__config__, 'MO_BASE', 1)

def kernel(bse, td_e, td_xy, e_GW, nstates=None, eris=None):
    '''static screening BSE excitation energies

    Returns:
        A list :  converged, number of states, excitation energies, eigenvectors
    '''
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

    matvec, diag = bse.gen_matvec(td_e, td_xy, e_GW, eris)
    
    size = nocc*nvir
    if not bse.TDA:
        size *= 2
    
    guess, nstates = bse.get_init_guess(nstates)
        
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
    conv, es, vs = eig(matvec, guess, precond, pick=pickeig,
                       tol=bse.conv_tol, max_cycle=bse.max_cycle,
                       max_space=bse.max_space, nroots=nroots, verbose=log)
        
    if bse.verbose >= logger.INFO:
        np.set_printoptions(threshold=nocc*nvir)
        logger.debug(bse, '  BSE excitation energies =\n%s', es.real)
        for n, en, vn, convn in zip(range(nroots), es, vs, conv):
            logger.info(bse, '  BSE root %d E = %.16g eV  conv = %s',
                        n, en*27.2114, convn)
        log.timer('BSE', *cput0) 
    
    return conv, nstates, es, vs
    
def matvec(bse, r, td_e=None, td_xy=None, e_GW=None, eris=None):
    '''matrix-vector multiplication'''
   
    nocc = bse.nocc
    nmo = bse.nmo
    nvir = nmo - nocc
    
    if eris is None: eris = bse.ao2mo()
    if td_e is None:
        td_e = bse._tdscf.e
    if td_xy is None:
        td_xy = bse._tdscf.xy
    if e_GW is None:
        e_GW = bse.e_GW

    

    nexc = nocc*nvir
    #X_ia + Y_ia
    td_z = np.sum(np.asarray(td_xy), axis=1).reshape(nexc,nocc,nvir)*math.sqrt(2)
    tdm_oo = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Loo)
    tdm_vv = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Lvv)
    tdm_ov = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Lov)
    
    e_GW_occ = e_GW[:nocc]
    e_GW_vir = e_GW[nocc:]
    
    r1 = r[:nocc*nvir].copy().reshape(nocc,nvir)
    #for A
    Hr1 = einsum('a,ia->ia', e_GW_vir, r1) - einsum('i,ia->ia', e_GW_occ, r1)
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
        Hr2 = einsum('a,ia->ia', e_GW_vir, r2) - einsum('i,ia->ia', e_GW_occ, r2)
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
    
def analyze(bse, verbose=None):
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
            #this is where I get '???'
    return wfnsym

class BSE(lib.StreamObject):
    '''spatial orbital static screening BSE

    Attributes:
        TDA : bool
            Whether to use the Tamm-Dancoff approximation to the BSE.  Default is True.
        singlet : bool
            Whether the excited state is a singlet or triplet.  Default is True.    
    Saved results:
        converged : bool
        e : float
            BSE eigenvalues (excitation energies)
        v :
            BSE eigenvectors 
        
    '''
     
    def __init__(self, mf, tdmf, e_GW, frozen=None, TDA=True, singlet=True, mo_coeff=None, mo_occ=None):
        assert(isinstance(mf, dft.rks.RKS) or isinstance(mf, dft.rks_symm.SymAdaptedRKS)) #or isinstance(mf, scf.rhf.RHF) 
        if mo_coeff  is None: mo_coeff  = mf.mo_coeff
        if mo_occ    is None: mo_occ    = mf.mo_occ
        
        self.mf = mf
        self.mol = mf.mol
        self._scf = mf
        self._tdscf = tdmf
        self.e_GW = e_GW
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
        self.converged = False
        self.e = None
        self.v = None
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._nocc = None
        self._nmo = None
        self._keys = set(self.__dict__.keys())

    def dump_flags(self, verbose=None):
        logger.info(self, '')
        logger.info(self, '******** %s ********', self.__class__)
        logger.info(self, 'nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen is not None:
            logger.info(self, 'frozen orbitals %s', self.frozen)
        logger.info(self, 'max_space = %d', self.max_space)
        logger.info(self, 'max_cycle = %d', self.max_cycle)
        logger.info(self, 'conv_tol = %s', self.conv_tol)
        logger.info(self, 'max_memory %d MB (current use %d MB)',
                    self.max_memory, lib.current_memory()[0])
        logger.info(self, 'BSE within the TDA = %s', self.TDA)
        logger.info(self, 'singlet = %s', self.singlet)
        return self
    
    def kernel(self, td_e=None, td_xy=None, e_GW=None, nstates=None, eris=None):
        if eris is None: 
            eris = self.ao2mo(self.mo_coeff)
        if td_e is None:
            td_e = self._tdscf.e
        if td_xy is None:
            td_xy = self._tdscf.xy
        if e_GW is None:
            e_GW = self.e_GW
        self.converged, self.nstates, self.es, self.vs = kernel(self, td_e, td_xy, e_GW, nstates, eris)
        return self.converged, self.nstates, self.es, self.vs

    matvec = matvec
    analyze = analyze
    analyze_wfnsym = analyze_wfnsym
    
    def gen_matvec(self, td_e=None, td_xy=None, e_GW=None, eris=None):
        if eris is None:
            eris = self.ao2mo()
        if td_e is None:
            td_e = self._tdscf.e
        if td_xy is None:
            td_xy = self._tdscf.xy
        if e_GW is None:
            e_GW = self.e_GW
        diag = self.get_diag(td_e, td_xy, e_GW, eris)
        matvec = lambda xs: [self.matvec(x, td_e, td_xy, e_GW, eris=eris) for x in xs]
        return matvec, diag

    def get_diag(self, td_e=None, td_xy=None, e_GW=None, eris=None):
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
        
        e_GW_occ = e_GW[:nocc]
        e_GW_vir = e_GW[nocc:]
        
        nexc = self._tdscf.nstates
        #X_ia + Y_ia
        td_z = np.sum(np.asarray(td_xy), axis=1).reshape(nexc,nocc,nvir)*math.sqrt(2)
        tdm_oo = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Loo)
        tdm_vv = einsum('via,Qia, Qpq->vpq', td_z, eris.Lov, eris.Lvv)
        
        diag = np.zeros((nocc,nvir))
        for i in range(nocc):
            for a in range(nvir):
                diag[i,a] += e_GW_vir[a] - e_GW_occ[i]
                diag[i,a] += 4*np.sum(tdm_oo[:,i,i]*tdm_vv[:,a,a]/td_e[:])
                diag[i,a] -= einsum('Q,Q', eris.Loo[:,i,i], eris.Lvv[:,a,a])
                if self.singlet:
                    diag[i,a] += 2*einsum('Q,Q', eris.Lov[:,i,a], eris.Lov[:,i,a])
        diag = diag.ravel()
        if self.TDA:
            return diag
        else: 
            return np.hstack((diag, -diag))
        
    def get_init_guess(self, nstates=None):
           
        if nstates is None: nstates = 1    
       
        # from pyscf import tdscf
        # guess = []
        # if self.TDA:
        #     td_ = tdscf.TDA(self.mf)
        #     td_.nstates = nstates
        #     td_.singlet=self.singlet
        #     td_.kernel()
        #     td_.analyze()
        #     for xy in td_.xy:
        #         x,y = xy
        #         g = x.ravel()
        #     guess.append(g)
        # else: 
        #     td_ = tdscf.RPA(self.mf)
        #     td_.nstates = nstates
        #     td_.singlet=self.singlet
        #     td_.kernel()
        #     td_.analyze()
        #     for xy in td_.xy:
        #         x,y = xy
        #         g = np.hstack((x.ravel(), y.ravel()))
        #     guess.append(g)
        
        from pyscf import tdscf
        guess = []
        td_ = tdscf.TDA(self.mf)
        td_.nstates = nstates
        td_.singlet=self.singlet
        td_.kernel()
        td_.analyze()
        for xy in td_.xy:
            x,y = xy
            g = x.ravel()
            if not self.TDA:
                g = np.hstack((g,g*0))
                # g = np.hstack((g,g*(-1.)))
            guess.append(g)
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
        with_df = df.DF(mygw.mol, auxbasis='def2-tzvp-ri' )
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
    from pyscf import gto
    from pyscf import scf
    
    mol = gto.Mole(unit='A')
    mol.verbose = 5
    # mol.atom = [['N',(0.0000, 0.0000, 0.0000)],
    #         ['N', (0.0000, 0.0000, 1.100779955618)]]
#formaldehyde, Thiel
    mol.atom = '''H      0.000000    0.934473    -0.588078
H      0.000000    -0.934473    -0.588078
C      0.000000    0.000000     0.000000
O      0.000000    0.000000     1.221104;'''
#butadiene, Thiel
#     mol.atom = '''
# H              1.080977       -2.558832         0.000000
#   H             -1.080977        2.558832         0.000000
#   H              2.103773       -1.017723         0.000000
#   H             -2.103773        1.017723         0.000000
#   H             -0.973565       -1.219040         0.000000
#   H              0.973565        1.219040         0.000000
#   C              0.000000        0.728881         0.000000
#   C              0.000000       -0.728881         0.000000
#   C              1.117962       -1.474815         0.000000
#   C             -1.117962        1.474815         0.000000;'''
    # mol.basis = 'cc-pVDZ'
    mol.basis = 'def2-tzvp'
    mol.cart = True
    mol.symmetry=True
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
    
    
    # bse = BSE(mf, td, e_GW, TDA=True, singlet=True)
    # conv, excitations, es, vs = bse.kernel(nstates = 1)
    # #es - 10.37315287680475
    
    bse = BSE(mf, td, e_GW, TDA=False, singlet=True)
    conv, excitations, es, vs = bse.kernel(nstates = 10)
    bse.analyze()
    #es[0]: 9.70241
    #es[1]: 10.36675902266208

    # bse = BSE(mf, td, e_GW, TDA=False, singlet=False)
    # conv, excitations, es, vs = bse.kernel(nstates = 4)
    #es[0]: 7.390942052862992 
    #es[1]: 8.560975213948101
    
    # bse = BSE(mf, td, e_GW, TDA=False, singlet=True)
    # conv, excitations, es, vs = bse.kernel(nstates = 2)
    # #es[0]: 9.702416355013936
    # #es[1]: 10.36675908660896
    
    ###########################
    
    #formaldehyde from Thiel set 
#     mol.atom = '''H      0.000000    0.934473    -0.588078
# H      0.000000    -0.934473    -0.588078
# C      0.000000    0.000000     0.000000
# O      0.000000    0.000000     1.221104;'''
#     mol.basis = 'def2-TZVP'
#     mol.cart = True
#     mol.symmetry = True
#     mol.build()
    
#     mf = dft.RKS(mol)
#     mf.xc = 'hf'
#     mf.kernel()

#     nocc = mol.nelectron//2
#     nmo = mf.mo_energy.size
#     nvir = nmo-nocc

#     td = tddft.dRPA(mf)
#     td.nstates = nocc*nvir
#     td.kernel()
 
#     singlet = True
    
#     #static part
#     static_bse = BSE(mf, td, e_GW, TDA=False, singlet=singlet)
#     converged, excitations, Omega_0_s, X_0_s  = static_bse.kernel(nstates = 10)
#     static_bse.analyze()