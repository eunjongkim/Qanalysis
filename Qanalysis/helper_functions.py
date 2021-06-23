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

_si_prefixes = {0: '',
           1: 'k',  2: 'M',  3: 'G',  4: 'T',  5: 'P',  6: 'E',  7: 'Z',  8: 'Y',  9: 'R',  10: 'Q',
          -1: 'm', -2: 'μ', -3: 'n', -4: 'p', -5: 'f', -6: 'a', -7: 'z', -8: 'y', -9: 'r', -10: 'q'
        }
_si_exponents = {value : key * 3 for (key, value) in _si_prefixes.items()}

def number_with_si_prefix(number):
    mantissa, exponent = f"{number:e}".split("e")
    prefixRange = int(exponent) // 3
    prefix = _si_prefixes.get(prefixRange, None)
    unitValue = float(mantissa) * 10 ** (int(exponent) % 3)
    return unitValue, prefix

def si_prefix_to_scaler(prefix: str):
    return 10 ** _si_exponents[prefix]