import torch
from typing import Optional, Union
from .parameter import BendingParameter
from .callback import BendingCallback, BendingCallbackException


_VALID_MODES = ["lowpass", "highpass", "bandpass", "bandstop"]
_VALID_METHODS = ["fft", "stft", "ir"]


def _sigmoid_edge(freq: torch.Tensor, edge: float, sharpness: float, rising: bool) -> torch.Tensor:
    z = (freq - edge) * sharpness
    if not rising:
        z = -z
    return torch.sigmoid(z)


def _resonance_bump(freq: torch.Tensor, center: float, resonance: float, sigma: float) -> torch.Tensor:
    return resonance * torch.exp(-((freq - center) ** 2) / (2. * sigma * sigma))


def _gain_curve(freq: torch.Tensor, mode: str, cutoff: float, bandwidth: float, resonance: float) -> torch.Tensor:
    """Real-valued gain over freq in [0, 1] (0 = DC, 1 = Nyquist). `resonance` sharpens the
    transition edge(s) and adds a peak bump there, mimicking an analog resonant filter."""
    sharpness = 8. + 40. * resonance
    sigma = 0.015 + 0.01 * resonance
    bw = max(bandwidth, 1e-4)
    if mode == "lowpass":
        gain = _sigmoid_edge(freq, cutoff, sharpness, rising=False)
        gain = gain + _resonance_bump(freq, cutoff, resonance, sigma)
    elif mode == "highpass":
        gain = _sigmoid_edge(freq, cutoff, sharpness, rising=True)
        gain = gain + _resonance_bump(freq, cutoff, resonance, sigma)
    elif mode == "bandpass":
        lo, hi = max(cutoff - bw / 2., 0.), min(cutoff + bw / 2., 1.)
        gain = _sigmoid_edge(freq, lo, sharpness, rising=True) * _sigmoid_edge(freq, hi, sharpness, rising=False)
        gain = gain + _resonance_bump(freq, lo, resonance, sigma) + _resonance_bump(freq, hi, resonance, sigma)
    else:
        lo, hi = max(cutoff - bw / 2., 0.), min(cutoff + bw / 2., 1.)
        band = _sigmoid_edge(freq, lo, sharpness, rising=True) * _sigmoid_edge(freq, hi, sharpness, rising=False)
        gain = 1. - band
        gain = gain + _resonance_bump(freq, lo, resonance, sigma) + _resonance_bump(freq, hi, resonance, sigma)
    return gain.clamp(min=0.)


class Filter(BendingCallback):
    """Filters the tensor along `dim` in the frequency domain: lowpass, highpass, bandpass or
    bandstop, with a resonance peak at the cutoff(s). `cutoff` and `bandwidth` are normalized in
    [0, 1] by default, where 0 is DC and 1 is Nyquist (half the length of `dim`). When `sr` is
    given, a value > 1 is instead read as Hz and converted against that sampling rate's Nyquist
    (sr / 2) — a value already in [0, 1] still means a normalized fraction, so both conventions
    can be used interchangeably once `sr` is set.
    `method` picks how the filter is realized: "fft" runs one global FFT / inverse-FFT over the
    whole axis (exact, non-causal, works on weights and activations); "ir" designs a windowed-sinc
    impulse response of length `ir_length` from the same mode/cutoff/resonance and applies it as a
    linear-phase FIR convolution (also weights and activations); "stft" filters framed, windowed
    chunks of the axis (`n_fft`/`hop_length`) and requires `dim` to be the tensor's last axis —
    meaningful for a genuine time/activation axis, not weights."""
    weight_compatible = True
    activation_compatible = True
    jit_compatible = False
    nntilde_compatible = False
    valid_modes = _VALID_MODES
    valid_methods = _VALID_METHODS
    controllable_params = {'cutoff': (float, 1.0), 'bandwidth': (float, 0.1), 'resonance': (float, 0.0)}
    _param_ui = {
        'cutoff': {
            'range': [0., 1.],
            'step': 0.001,
            'description': "Normalized cutoff frequency: 0 = DC, 1 = Nyquist (half the axis length). "
                            "If `sr` is set, a value > 1 is read as Hz instead.",
        },
        'bandwidth': {
            'range': [0., 1.],
            'step': 0.001,
            'description': "Band width around the cutoff. Only used in bandpass/bandstop modes. "
                            "If `sr` is set, a value > 1 is read as Hz instead.",
        },
        'resonance': {
            'range': [0., 4.],
            'step': 0.01,
            'description': "Emphasis at the cutoff(s): sharper transition plus a boosted peak.",
        },
    }
    _extra_init_params = {
        "dim": {"type": "int", "default": -1, "required": True, "label": "dim (axis)",
                "description": "The tensor axis that is filtered."},
        "mode": {"type": "str", "default": "lowpass", "required": False, "choices": _VALID_MODES,
                 "description": "lowpass / highpass / bandpass / bandstop."},
        "method": {"type": "str", "default": "fft", "required": False, "choices": _VALID_METHODS,
                   "description": "fft: exact global spectral filter. ir: windowed-sinc FIR convolution. "
                                   "stft: framed/windowed spectral filter (dim must be the last axis)."},
        "n_fft": {"type": "int", "default": 1024, "required": False, "label": "n_fft (stft)",
                  "description": "STFT frame size. Only used when method='stft'."},
        "hop_length": {"type": "int", "default": None, "required": False, "label": "hop (stft)",
                       "description": "STFT hop size. Only used when method='stft'; defaults to n_fft // 4."},
        "ir_length": {"type": "int", "default": 129, "required": False, "label": "IR length",
                      "description": "Impulse-response length in samples. Only used when method='ir'."},
        "sr": {"type": "int", "default": -1, "required": False, "label": "sample rate",
               "description": "When set (>= 0), cutoff/bandwidth are read in Hz instead of normalized [0, 1] "
                                "(converted against this sampling rate's Nyquist, sr / 2). -1 = unset."},
    }

    def __init__(self, dim: int, mode: str = "lowpass", method: str = "fft",
                 cutoff: Union[float, BendingParameter] = 1.0,
                 bandwidth: Union[float, BendingParameter] = 0.1,
                 resonance: Union[float, BendingParameter] = 0.0,
                 n_fft: int = 1024, hop_length: Optional[int] = None, ir_length: int = 129,
                 sr: int = -1):
        super().__init__(cutoff=cutoff, bandwidth=bandwidth, resonance=resonance)
        assert mode in self.valid_modes, f"mode must be one of {self.valid_modes}, got {mode}"
        assert method in self.valid_methods, f"method must be one of {self.valid_methods}, got {method}"
        self.mode = mode
        self.method = method
        self.register_buffer('dim', torch.tensor(dim).int())
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length) if hop_length is not None else max(self.n_fft // 4, 1)
        ir_length = int(ir_length)
        self.ir_length = ir_length if ir_length % 2 == 1 else ir_length + 1
        sr = int(sr) if sr is not None else -1
        self.sr = None if sr < 0 else sr

    def __repr__(self):
        return f"Filter(dim={int(self.dim)}, mode={self.mode}, method={self.method}, sr={self.sr})"

    def _resolve_dim(self, ndim: int) -> int:
        d = int(self.dim)
        return ndim + d if d < 0 else d

    def _normalize_freq(self, value: float) -> float:
        """Converts a Hz value to a normalized [0, 1] fraction of Nyquist when `sr` is set.

        A value already in [0, 1] is assumed to already be normalized and is passed through
        unchanged — only a value > 1 (unambiguously not a normalized fraction) is read as Hz
        and converted. Without `sr`, everything passes through unchanged, as before."""
        if self.sr is None or value <= 1.:
            return value
        return value / (self.sr / 2.)

    # ------------------------------------------------------------------ fft
    def _fft_filter(self, x: torch.Tensor, cutoff: float, bandwidth: float, resonance: float) -> torch.Tensor:
        dim = self._resolve_dim(x.ndim)
        if x.ndim == 0 or dim >= x.ndim:
            return x
        L = x.shape[dim]
        if L < 2:
            return x
        Xf = torch.fft.rfft(x, dim=dim)
        n_bins = Xf.shape[dim]
        freq = torch.linspace(0., 1., n_bins, device=x.device, dtype=torch.float32)
        gain = _gain_curve(freq, self.mode, cutoff, bandwidth, resonance)
        shape = [1] * x.ndim
        shape[dim] = n_bins
        gain = gain.reshape(shape).to(x.device)
        return torch.fft.irfft(Xf * gain, n=L, dim=dim)

    # ------------------------------------------------------------------- ir
    def _design_ir(self, cutoff: float, bandwidth: float, resonance: float, device, dtype) -> torch.Tensor:
        N = self.ir_length
        n_bins = N // 2 + 1
        freq = torch.linspace(0., 1., n_bins, device=device, dtype=torch.float32)
        gain = _gain_curve(freq, self.mode, cutoff, bandwidth, resonance)
        spectrum = torch.complex(gain, torch.zeros_like(gain))
        ir = torch.fft.irfft(spectrum, n=N)
        ir = torch.fft.fftshift(ir)
        window = torch.hann_window(N, periodic=False, device=device, dtype=ir.dtype)
        return (ir * window).to(dtype)

    def _ir_filter(self, x: torch.Tensor, cutoff: float, bandwidth: float, resonance: float) -> torch.Tensor:
        dim = self._resolve_dim(x.ndim)
        if x.ndim == 0 or dim >= x.ndim:
            return x
        L = x.shape[dim]
        if L < 2:
            return x
        ir = self._design_ir(cutoff, bandwidth, resonance, x.device, x.dtype)
        N = ir.shape[0]
        n_conv = L + N - 1
        Xf = torch.fft.rfft(x, n=n_conv, dim=dim)
        ir_shape = [1] * x.ndim
        ir_shape[dim] = N
        Hf = torch.fft.rfft(ir.reshape(ir_shape), n=n_conv, dim=dim)
        y = torch.fft.irfft(Xf * Hf, n=n_conv, dim=dim)
        start = (N - 1) // 2
        return y.narrow(dim, start, L)

    # ----------------------------------------------------------------- stft
    def _stft_filter(self, x: torch.Tensor, cutoff: float, bandwidth: float, resonance: float) -> torch.Tensor:
        dim = self._resolve_dim(x.ndim)
        if dim != x.ndim - 1:
            raise BendingCallbackException(
                "Filter(method='stft') requires dim to be the tensor's last axis, got dim=%d of %d" % (dim, x.ndim))
        orig_shape = x.shape
        L = orig_shape[-1]
        if L < 2:
            return x
        x_flat = x.reshape(-1, L)
        window = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        Xf = torch.stft(x_flat, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                         return_complex=True, center=True)
        n_bins = Xf.shape[1]
        freq = torch.linspace(0., 1., n_bins, device=x.device, dtype=torch.float32)
        gain = _gain_curve(freq, self.mode, cutoff, bandwidth, resonance).to(x.device)
        Xf = Xf * gain.reshape(1, n_bins, 1)
        y = torch.istft(Xf, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                         length=L, center=True)
        return y.reshape(orig_shape)

    def _filter(self, x: torch.Tensor, cutoff: float, bandwidth: float, resonance: float) -> torch.Tensor:
        if self.method == "fft":
            return self._fft_filter(x, cutoff, bandwidth, resonance)
        elif self.method == "ir":
            return self._ir_filter(x, cutoff, bandwidth, resonance)
        else:
            return self._stft_filter(x, cutoff, bandwidth, resonance)

    def apply_to_param(self, idx: int, param: torch.nn.Parameter, cache: torch.Tensor) -> None:
        cutoff = self.get('cutoff')
        if cutoff is None:
            return
        bandwidth = self.get('bandwidth')
        resonance = self.get('resonance')
        with torch.no_grad():
            out = self._filter(cache, self._normalize_freq(float(cutoff)),
                                self._normalize_freq(float(bandwidth) if bandwidth is not None else 0.1),
                                float(resonance) if resonance is not None else 0.0)
            param.set_(out)

    def bend_input(self, x: torch.Tensor, cutoff: Optional[torch.Tensor] = None,
                    bandwidth: Optional[torch.Tensor] = None, resonance: Optional[torch.Tensor] = None,
                    name: Optional[str] = None):
        if cutoff is None:
            return x
        return self._filter(x, self._normalize_freq(float(cutoff)),
                             self._normalize_freq(float(bandwidth) if bandwidth is not None else 0.1),
                             float(resonance) if resonance is not None else 0.0)
