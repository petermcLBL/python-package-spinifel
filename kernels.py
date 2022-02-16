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

def core_problem_convolution_kernel(xp, ugrid, M, F_ugrid_conv_, M_ups):
    # Upsample
    ugrid_ups = xp.zeros((M_ups,) * 3, dtype=xp.complex128)
    ugrid_ups[:M, :M, :M] = ugrid
    # Convolution = Fourier multiplication
    F_ugrid_ups = fftn(xp.fft.ifftshift(ugrid_ups))
    F_ugrid_conv_out_ups = F_ugrid_ups * F_ugrid_conv_
    # Kludge needed here, for NumPy (not CuPy) array, but nowhere else.
    if (F_ugrid_conv_out_ups.flags.c_contiguous):
        ugrid_conv_out_ups = xp.fft.fftshift(ifftn(F_ugrid_conv_out_ups))
    else:
        F_ugrid_conv_out_ups_C = F_ugrid_conv_out_ups.astype(xp.complex128,
                                                             order='C')
        ugrid_conv_out_ups = xp.fft.fftshift(ifftn(F_ugrid_conv_out_ups_C))
    # Downsample
    ugrid_conv_out = ugrid_conv_out_ups[:M, :M, :M].real
    return ugrid_conv_out
