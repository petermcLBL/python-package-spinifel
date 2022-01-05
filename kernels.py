from fftx.fft import *

def step_phase_kernel(xp, rho_, amp_mask_, amplitudes_):
    rho_hat_ = fftn(rho_)
    phases_ = xp.angle(rho_hat_)
    rho_hat_mod_ = xp.where(
        amp_mask_,
        amplitudes_ * xp.exp(1j*phases_),
        rho_hat_)
    rho_mod_ = ifftn(rho_hat_mod_).real
    return rho_mod_
