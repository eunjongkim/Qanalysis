import numpy as np

def min_max_delta_to_array(min_, max_, delta_):
    return np.arange(min_, max_ + delta_ / 2, delta_)

def volt_p_to_dBm(vp, Z0=50):
    """
    Convert peak voltage `vp` to dBm, assuming impedance `Z0`
    """
    P = vp ** 2 / (2 * Z0)
    return watt_to_dBm(P)

def volt_pp_to_dBm(vpp, Z0=50):
    """
    Convert peak-to-peak voltage `vpp` to dBm, assuming impedance `Z0`
    """
    return volt_p_to_dBm(vpp / 2)

def watt_to_dBm(p):
    """
    Convert power in units of Watt into dBm
    """
    return 10 * np.log10(p / 1e-3)

def dBm_to_watt(p_dBm):
    """
    Convert power in units of dBm into Watt
    """
    return 1e-3 * 10 ** (p_dBm / 10)

def dBm_to_volt_p(p_dBm, Z0=50):
    """
    Convert power in units of dBm into peak voltage
    """
    return np.sqrt(2 * Z0 * dBm_to_watt(p_dBm))

def dBm_to_volt_pp(p_dBm, Z0=50):
    """
    Convert power in units of dBm into peak-to-peak voltage
    """
    return 2 * dBm_to_volt_p(p_dBm, Z0=50)

def pow_ratio_to_dB(r):
    """
    Convert power ratio `r` into dB number
    """
    return 10 * np.log10(r)