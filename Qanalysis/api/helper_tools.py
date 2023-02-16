import numpy as np
from dataclasses import dataclass, field
from scipy.signal import windows

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
_si_exponents = {value: key * 3 for (key, value) in _si_prefixes.items()}

def number_with_si_prefix(number: float) -> tuple[float, str]:
    mantissa, exponent = f"{number:e}".split("e")
    prefixRange = int(exponent) // 3
    prefix = _si_prefixes.get(prefixRange, None)
    unitValue = float(mantissa) * 10 ** (int(exponent) % 3)
    return unitValue, prefix

def si_prefix_to_scaler(prefix: str):
    return 10 ** _si_exponents[prefix]


def get_envelope(s: np.ndarray, t: np.ndarray, f: float):
    """
    Find the envelope of an oscillating real-valued signal `s` as a function of time `t` by demodulating at the
    frequency specified by `f` followed by low-pass filtering at cutoff of half the specified frequency.
    """
    # time step
    dt: float = t[1] - t[0]

    # perform manual demodulation to get envelope
    I: np.ndarray = s * np.cos(2 * np.pi * f * t)
    Q: np.ndarray = - s * np.sin(2 * np.pi * f * t)

    # extract envelope by low-pass filtering at cutoff of f / 2
    _window = int(1 / (f * dt))
    _hann = windows.hann(_window * 2, sym=True)
    _hann = _hann / np.sum(_hann)
    envComplex = np.convolve(I + 1j * Q, _hann, 'same') * 2
    return envComplex[:len(s)]


@dataclass(eq = True)
class ScientificNumber:
    """
    Dataclass for representing a number (with errorbar) with a unit.
    """
    number: float
    error: float = None
    unit_prefix: str = field(default_factory = str)
    base_unit: str = field(default_factory = str)
    numeric_scale: int = field(repr = False, default = 3)

    def __post_init__(self) -> None:
        mantissa, exponent = f"{(self.number * self.scaler):e}".split("e")
        prefixRange = int(exponent) // 3

        self.unit_prefix = _si_prefixes.get(prefixRange, None)
        self.number = float(mantissa) * 10 ** (int(exponent) % 3)

        if self.error is not None:
            self.error /= self.scaler

    @property
    def scaler(self) -> float:
        return 10 ** _si_exponents[self.unit_prefix]

    @property
    def unit(self) -> str:
        """
        Unit of the quantity including the prefix
        """
        return self.unit_prefix + self.base_unit

    def __str__(self) -> str:
        if self.error is None:
            return f"{self.number:.{self.numeric_scale}f} {self.unit}"
        else:
            return (
                f"{self.number:.{self.numeric_scale}f}" + 
                f" ± {self.error:.{self.numeric_scale}f} {self.unit}"
            )