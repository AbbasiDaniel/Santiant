#!/usr/bin/env python3
"""Sentiant emotion model: 8 emotions + 4 metrics (stability, intensity, burstiness, wavelength).
Automatically generates a complete emotion_model.h for the bracelet.
pip install numpy scipy torch
python emotion_training.py
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple
import numpy as np
import torch
from scipy.signal import lfilter
from scipy.special import ndtri
from torch import nn

SEED = 7
PEOPLE = 24
DAYS = 6
HOURS_PER_DAY = 12
EPOCH_SEC = 5
WINDOW_SEC = 900
WINDOW_EPOCHS = WINDOW_SEC // EPOCH_SEC

# ---------- 8 emotions ----------
EMOTIONS = (
    "calm",      # 0
    "tense",     # 1
    "upbeat",    # 2
    "subdued",   # 3
    "happy",     # 4
    "sad",       # 5
    "angry",     # 6
    "tired",     # 7
)
EMOTION_SHARE = (0.22, 0.12, 0.12, 0.10, 0.14, 0.10, 0.10, 0.10)

HR, MOTION, PITCH, ENERGY = range(4)
SIGNALS = ("hr", "motion", "pitch", "energy")
METRICS = ("stability", "intensity", "burstiness", "wavelength")
FEATURES = tuple(f"{s}_{m}" for s in SIGNALS for m in METRICS) + (
    "voiced_fraction",
    "voice_available",
)
VOICED_FRACTION, VOICE_FLAG = 16, 17
PRIOR = np.array([70.0, 30.0, 160.0, 30.0])
MEAN_FLOOR = np.array([1.0, 10.0, 10.0, 1.0])
MIN_EPOCHS = np.array([90, 90, 6, 6])
ACTIVITY_GATE_MG = 150.0
PEAK_SIGMA = 0.5
BASELINE_ALPHA_MIN = 0.01
MIN_LABEL_PURITY = 0.8
STATUS_OK, STATUS_LOW_CONFIDENCE, STATUS_HIGH_MOTION, STATUS_NO_DATA = range(4)
NO_ESTIMATE = 255

HEART_INPUTS = list(range(0, 4))
MOTION_INPUTS = list(range(4, 8))
VOICE_INPUTS = list(range(8, 18))
HEART_SIZE, MOTION_SIZE, VOICE_SIZE = 6, 6, 8
DEEP_SIZES = (16, 12, 8)
INPUT_LIMIT = 5.0
TARGET_ACCURACY = 0.80
MIN_KEPT_SHARE = 0.18

# 8 rows × 9 physiological effect parameters
# [hr_shift, hr_wobble, hr_period, hr_spikes, motion_gain, pitch_ratio, pitch_jitter, energy_shift, talk_gain]
EFFECTS = np.array(
    [
        # calm
        [  2.0, 1.5, 180.0,  2.0, 1.00, 1.00, 0.040,  0.0, 1.00],
        # tense
        [ 16.0, 3.0,  60.0, 20.0, 1.50, 1.09, 0.070,  3.0, 1.10],
        # upbeat
        [ 10.0, 2.5, 110.0,  6.0, 1.80, 1.07, 0.120,  5.0, 1.60],
        # subdued
        [ -4.0, 0.8, 240.0,  1.0, 0.60, 0.95, 0.025, -4.0, 0.50],
        # happy
        [  6.0, 2.0, 130.0,  4.0, 1.40, 1.12, 0.090,  6.0, 1.80],
        # sad
        [ -2.0, 1.2, 220.0,  3.0, 0.70, 0.92, 0.030, -5.0, 0.40],
        # angry
        [ 18.0, 3.5,  70.0, 25.0, 2.20, 1.15, 0.100,  8.0, 1.40],
        # tired
        [ -6.0, 0.6, 260.0,  1.5, 0.50, 0.90, 0.020, -6.0, 0.30],
    ]
)


def ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    return lfilter([1.0], [1.0, -phi], rng.standard_normal(n) * sd * math.sqrt(1.0 - phi**2))


def random_blocks(
    rng: np.random.Generator, n: int, count: int, shortest: int, longest: int
) -> Iterator[slice]:
    for _ in range(count):
        start = int(rng.integers(0, n))
        yield slice(start, start + int(rng.integers(shortest, longest)))


def simulate_person(rng: np.random.Generator) -> Dict[str, np.ndarray]:
    per_min = 60 // EPOCH_SEC
    n = DAYS * HOURS_PER_DAY * 60 * per_min
    n_classes = len(EMOTIONS)
    hr_rest = float(np.clip(rng.normal(68.0, 7.0), 50.0, 90.0))
    f0 = rng.normal(115.0, 15.0) if rng.random() < 0.5 else rng.normal(205.0, 25.0)
    energy_rest = rng.normal(30.0, 3.0)
    talk_rest = float(np.clip(rng.beta(2.0, 5.0), 0.05, 0.6))
    motion_rest = rng.lognormal(math.log(25.0), 0.4)
    mic_off_share = rng.beta(1.5, 4.0)
    response = rng.lognormal(0.0, 0.35, size=EFFECTS.shape)
    n_segments = n // (20 * per_min) + 1
    seg_class = rng.choice(n_classes, n_segments, p=EMOTION_SHARE)
    seg_level = np.where(
        seg_class == 0, rng.uniform(0.05, 0.3, n_segments), rng.uniform(0.3, 1.0, n_segments)
    )
    seg_len = rng.integers(20 * per_min, 120 * per_min, n_segments)
    state = np.repeat(seg_class, seg_len)[:n]
    level = np.repeat(seg_level, seg_len)[:n]
    wrong = rng.random(n_segments) < 0.05
    seg_label = np.where(
        wrong, (seg_class + rng.integers(1, n_classes, n_segments)) % n_classes, seg_class
    )
    label = np.repeat(seg_label, seg_len)[:n]
    neutral = EFFECTS[0]
    personal = neutral + (EFFECTS - neutral) * response
    (
        hr_shift,
        hr_wobble,
        hr_period,
        hr_spikes,
        motion_gain,
        pitch_ratio,
        pitch_jitter,
        energy_shift,
        talk_gain,
    ) = (neutral + (personal[state] - neutral) * level[:, None]).T
    t = np.arange(n) * EPOCH_SEC
    drift = 3.0 * np.sin(2.0 * np.pi * t / (HOURS_PER_DAY * 3600.0))
    phase = np.cumsum(2.0 * np.pi * EPOCH_SEC / np.maximum(hr_period, 30.0))
    onsets = rng.random(n) < np.maximum(hr_spikes, 0.0) * EPOCH_SEC / 3600.0
    spikes = lfilter([1.0], [1.0, -math.exp(-EPOCH_SEC / 20.0)], onsets * rng.exponential(8.0, n))
    hr = (
        hr_rest
        + drift
        + hr_shift
        + np.maximum(hr_wobble, 0.2) * np.sin(phase)
        + ar1(rng, n, 0.9, 1.5)
        + spikes
    )
    motion = (
        motion_rest * np.maximum(motion_gain, 0.1) * np.exp(ar1(rng, n, 0.9, 0.5)) + 6.0 * spikes
    )
    exercising = np.zeros(n, dtype=bool)
    for block in random_blocks(rng, n, rng.poisson(1.5 * DAYS), 15 * per_min, 45 * per_min):
        exercising[block] = True
        hr[block] += rng.uniform(30.0, 60.0)
        motion[block] += rng.uniform(200.0, 500.0)
    talk = np.clip(talk_rest * np.maximum(talk_gain, 0.1), 0.01, 0.9)
    voiced = ar1(rng, n, 0.8, 1.0) < ndtri(talk)
    hour = 60 * per_min
    mic_on = np.repeat(rng.random(n // hour + 1) > mic_off_share, hour)[:n]
    heard = voiced & mic_on
    pitch = f0 * pitch_ratio * (1.0 + np.maximum(pitch_jitter, 0.005) * ar1(rng, n, 0.6, 1.0))
    energy = energy_rest + energy_shift + ar1(rng, n, 0.6, 2.0)
    pitch[~heard] = np.nan
    energy[~heard] = np.nan
    hr[ar1(rng, n, 0.8, 1.0) > np.where(exercising, 1.0, 2.2)] = np.nan
    for block in random_blocks(rng, n, rng.poisson(0.5 * DAYS), 30 * per_min, 120 * per_min):
        hr[block] = motion[block] = pitch[block] = energy[block] = np.nan
        mic_on[block] = False
    signals = np.stack((hr, motion, pitch, energy), axis=1).astype(np.float32)
    return dict(signals=signals, mic_on=mic_on, label=label, level=level)


def peak_interval(x: np.ndarray, mean: float, std: float) -> float:
    here, before, after = x[1:-1], x[:-2], x[2:]
    peaks = np.flatnonzero((here > before) & (here >= after) & (here > mean + PEAK_SIGMA * std))
    if peaks.size < 2:
        return float(WINDOW_SEC)
    return (peaks[-1] - peaks[0]) / (peaks.size - 1) * EPOCH_SEC


@dataclass
class Window:
    features: np.ndarray
    means: np.ndarray
    usable: np.ndarray
    status: int


def window_features(segment: np.ndarray, mic_on: np.ndarray, baseline: np.ndarray) -> Window:
    x = segment.astype(np.float64)
    x[~np.isfinite(x)] = np.nan
    x[np.ix_(~mic_on, (PITCH, ENERGY))] = np.nan
    valid = ~np.isnan(x)
    count = valid.sum(axis=0)
    safe = np.maximum(count, 1)
    mean = np.where(valid, x, 0.0).sum(axis=0) / safe
    std = np.sqrt((np.where(valid, x - mean, 0.0) ** 2).sum(axis=0) / safe)
    usable = count >= MIN_EPOCHS
    voice_ok = bool(usable[PITCH] and usable[ENERGY])
    features = np.full(len(FEATURES), np.nan)
    for c in range(len(SIGNALS)):
        if not usable[c] or (c in (PITCH, ENERGY) and not voice_ok):
            continue
        first = c * len(METRICS)
        features[first : first + len(METRICS)] = (
            std[c],
            mean[c] - baseline[c],
            std[c] / max(abs(mean[c]), MEAN_FLOOR[c]),
            peak_interval(x[:, c], mean[c], std[c]),
        )
    if voice_ok:
        features[VOICED_FRACTION] = count[PITCH] / max(int(mic_on.sum()), 1)
        features[VOICE_FLAG] = float(voice_ok)
    if not (usable[HR] and usable[MOTION]):
        status = STATUS_NO_DATA
    elif mean[MOTION] >= ACTIVITY_GATE_MG:
        status = STATUS_HIGH_MOTION
    else:
        status = STATUS_OK
    return Window(features, mean, usable, status)


class Baseline:
    def __init__(self) -> None:
        self.value = PRIOR.copy()
        self.count = np.zeros(len(SIGNALS), dtype=np.int64)

    def update(self, window: Window) -> None:
        if not window.usable[MOTION] or window.means[MOTION] >= ACTIVITY_GATE_MG:
            return
        for c in np.flatnonzero(window.usable):
            self.count[c] += 1
            alpha = max(1.0 / (self.count[c] + 1.0), BASELINE_ALPHA_MIN)
            self.value[c] += alpha * (window.means[c] - self.value[c])


def person_windows(person: Dict[str, np.ndarray]) -> Iterator[Tuple[int, Window]]:
    baseline = Baseline()
    for start in range(0, len(person["signals"]) - WINDOW_EPOCHS + 1, WINDOW_EPOCHS):
        end = start + WINDOW_EPOCHS
        segment, mic_on = person["signals"][start:end], person["mic_on"][start:end]
        window = window_features(segment, mic_on, baseline.value)
        yield start, window
        baseline.update(window)


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    level: np.ndarray
    person: np.ndarray
    status: np.ndarray

    def select(self, mask: np.ndarray) -> "Dataset":
        return Dataset(
            self.X[mask], self.y[mask], self.level[mask], self.person[mask], self.status[mask]
        )


def build_dataset(people: Sequence[Dict[str, np.ndarray]]) -> Dataset:
    X, y, level, who, status = [], [], [], [], []
    for index, person in enumerate(people):
        for start, window in person_windows(person):
            end = start + WINDOW_EPOCHS
            values, counts = np.unique(person["label"][start:end], return_counts=True)
            pure = counts.max() >= MIN_LABEL_PURITY * WINDOW_EPOCHS
            X.append(window.features)
            y.append(int(values[counts.argmax()]) if pure else -1)
            level.append(float(person["level"][start:end].mean()))
            who.append(index)
            status.append(window.status)
    return Dataset(
        np.array(X),
        np.array(y, dtype=np.int64),
        np.array(level, dtype=np.float32),
        np.array(who, dtype=np.int64),
        np.array(status, dtype=np.int64),
    )


def symlog(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Scaler":
        f = symlog(X)
        valid = ~np.isnan(f)
        safe = np.maximum(valid.sum(axis=0), 1)
        mean = np.where(valid, f, 0.0).sum(axis=0) / safe
        std = np.sqrt((np.where(valid, f - mean, 0.0) ** 2).sum(axis=0) / safe)
        return cls(mean.astype(np.float32), np.where(std > 1e-6, std, 1.0).astype(np.float32))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (symlog(X) - self.mean.astype(np.float64)) / self.std.astype(np.float64)

    def inputs(self, X: np.ndarray) -> np.ndarray:
        z = np.clip(self.transform(X), -INPUT_LIMIT, INPUT_LIMIT)
        return np.nan_to_num(z).astype(np.float32)


class FusionNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heart = nn.Linear(len(HEART_INPUTS), HEART_SIZE)
        self.motion = nn.Linear(len(MOTION_INPUTS), MOTION_SIZE)
        self.voice = nn.Linear(len(VOICE_INPUTS), VOICE_SIZE)
        sizes = (HEART_SIZE + MOTION_SIZE + VOICE_SIZE, *DEEP_SIZES)
        self.deep = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.emotion = nn.Linear(sizes[-1], len(EMOTIONS))
        self.intensity = nn.Linear(sizes[-1], 1)

    def states(self, z: torch.Tensor) -> List[torch.Tensor]:
        heart, motion = self.heart(z[:, HEART_INPUTS]), self.motion(z[:, MOTION_INPUTS])
        encoded = torch.cat((heart, motion, self.voice(z[:, VOICE_INPUTS])), dim=1)
        states = [torch.relu(encoded)]
        for layer in self.deep:
            states.append(torch.relu(layer(states[-1])))
        return states

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        last = self.states(z)[-1]
        return self.emotion(last), self.intensity(last).squeeze(-1)


def hide_voice(z: torch.Tensor, hide: torch.Tensor, flag_off: float) -> torch.Tensor:
    z = z.clone()
    z[:, VOICE_INPUTS[:-1]] = z[:, VOICE_INPUTS[:-1]].masked_fill(hide[:, None], 0.0)
    z[:, VOICE_FLAG] = z[:, VOICE_FLAG].masked_fill(hide, flag_off)
    return z


def train(train_set: Dataset, val_set: Dataset, scaler: Scaler) -> FusionNet:
    torch.manual_seed(SEED)
    shuffle = torch.Generator().manual_seed(SEED)
    x, y = torch.as_tensor(scaler.inputs(train_set.X)), torch.as_tensor(train_set.y)
    level = torch.as_tensor(train_set.level)
    val_x, val_y = torch.as_tensor(scaler.inputs(val_set.X)), torch.as_tensor(val_set.y)
    val_level = torch.as_tensor(val_set.level)
    counts = np.maximum(np.bincount(train_set.y, minlength=len(EMOTIONS)), 1)
    weight = torch.as_tensor(len(train_set.y) / (len(EMOTIONS) * counts), dtype=torch.float32)
    flag_off = float(-scaler.mean[VOICE_FLAG] / scaler.std[VOICE_FLAG])
    net = FusionNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.002, weight_decay=0.001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

    def loss_of(inputs: torch.Tensor, labels: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        logits, predicted = net(inputs)
        emotion_loss = nn.functional.cross_entropy(logits, labels, weight=weight)
        return emotion_loss + 2.0 * nn.functional.mse_loss(predicted, levels)

    best, best_state, stale = math.inf, None, 0
    for epoch in range(120):
        net.train()
        for batch in torch.randperm(len(x), generator=shuffle).split(64):
            hide = torch.rand(len(batch), generator=shuffle) < 0.2
            optimizer.zero_grad()
            loss_of(hide_voice(x[batch], hide, flag_off), y[batch], level[batch]).backward()
            optimizer.step()
            schedule.step()
        net.eval()
        with torch.no_grad():
            loss = float(loss_of(val_x, val_y, val_level))
            if loss < best - 1e-4:
                best, best_state, stale = loss, copy.deepcopy(net.state_dict()), 0
            else:
                stale += 1
            if stale >= 40:
                break
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1}, val loss {loss:.4f}")
    net.load_state_dict(best_state)
    net.eval()
    return net


@dataclass
class Layer:
    weights: np.ndarray
    bias: np.ndarray
    mult: int
    shift: int
    in_zero: int
    out_zero: int
    out_min: int
    apply_relu: bool = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        total = (x.astype(np.int64) - self.in_zero) @ self.weights.T + self.bias
        rescaled = (total * self.mult + (1 << (self.shift - 1))) >> self.shift
        out = rescaled + self.out_zero
        if self.apply_relu:
            out = np.maximum(out, self.out_zero)
        return np.clip(out, self.out_min, 127).astype(np.int64)


def quantize_layer(
    layer: nn.Linear,
    in_scale: float,
    in_zero: int,
    out_scale: float,
    out_zero: int,
    out_min: int,
    apply_relu: bool = False,
) -> Layer:
    weights = layer.weight.detach().numpy().astype(np.float64)
    bias = layer.bias.detach().numpy().astype(np.float64)
    w_scale = max(np.abs(weights).max() / 127.0, 1e-9)
    b_scale = in_scale * w_scale
    if np.abs(bias).max() > b_scale * (1 << 30):
        w_scale = max(w_scale, np.abs(bias).max() / (in_scale * (1 << 30)))
        b_scale = in_scale * w_scale
    q_weights = np.clip(np.rint(weights / w_scale), -127, 127).astype(np.int64)
    q_bias = np.rint(bias / b_scale).astype(np.int64)
    real_mult = (in_scale * w_scale) / out_scale
    mantissa, exponent = math.frexp(real_mult)
    mult = int(round(mantissa * (1 << 31)))
    if mult == (1 << 31):
        mult //= 2
        exponent += 1
    shift = 31 - exponent
    while shift < 0:
        mult = max(mult // 2, 1)
        shift += 1
    return Layer(
        q_weights, q_bias, mult, shift, in_zero, out_zero, out_min, apply_relu=apply_relu
    )


@dataclass
class QuantizedNet:
    scaler: Scaler
    input_scale: float
    heart: Layer
    motion: Layer
    voice: Layer
    deep: List[Layer]
    emotion: Layer
    intensity: Layer
    logit_scale: float
    level_scale: float
    temperature: float = 1.0
    min_confidence: int = 0

    def quantize_inputs(self, X: np.ndarray) -> np.ndarray:
        q = np.floor(self.scaler.transform(X) / self.input_scale + 0.5)
        return np.nan_to_num(np.clip(q, -127, 127)).astype(np.int64)

    def forward(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        heart = self.heart(q[:, HEART_INPUTS])
        motion = self.motion(q[:, MOTION_INPUTS])
        voice = self.voice(q[:, VOICE_INPUTS])
        x = np.hstack((heart, motion, voice))
        for layer in self.deep:
            x = layer(x)
        return self.emotion(x), self.intensity(x)

    def confidence(self, logits: np.ndarray) -> np.ndarray:
        gaps = (logits - logits.max(axis=1, keepdims=True)).astype(np.float64)
        gaps = gaps * self.logit_scale / self.temperature
        exp_gaps = np.exp(np.clip(gaps, -50.0, 0.0))
        return np.floor(100.0 / exp_gaps.sum(axis=1) + 0.5).astype(np.int64)

    def predict_quantized(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits, level = self.forward(q)
        percent = np.clip(
            np.floor(level[:, 0].astype(np.float64) * self.level_scale * 100.0 + 0.5), 0, 100
        )
        return logits.argmax(axis=1), self.confidence(logits), percent.astype(np.int64)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.predict_quantized(self.quantize_inputs(X))


def quantize(net: FusionNet, train_set: Dataset, scaler: Scaler) -> QuantizedNet:
    z = torch.as_tensor(scaler.inputs(train_set.X))
    flag_off = float(-scaler.mean[VOICE_FLAG] / scaler.std[VOICE_FLAG])
    z = torch.cat((z, hide_voice(z, torch.ones(len(z), dtype=torch.bool), flag_off)))

    with torch.no_grad():
        states = [s.numpy() for s in net.states(z)]
        last = net.states(z)[-1]
        logits = net.emotion(last).numpy()
        level = net.intensity(last).numpy()

    input_scale = float(np.float32(INPUT_LIMIT / 127.0))

    def act_scale(act: np.ndarray) -> float:
        p = float(np.percentile(act, 99.9))
        return max(p, 1e-3) / 127.0

    scale0 = act_scale(states[0])

    heart = quantize_layer(net.heart, input_scale, 0, scale0, 0, 0, apply_relu=True)
    motion = quantize_layer(net.motion, input_scale, 0, scale0, 0, 0, apply_relu=True)
    voice = quantize_layer(net.voice, input_scale, 0, scale0, 0, 0, apply_relu=True)

    deep: List[Layer] = []
    prev_scale = scale0
    for layer, state in zip(net.deep, states[1:]):
        out_scale = act_scale(state)
        deep.append(quantize_layer(layer, prev_scale, 0, out_scale, 0, 0, apply_relu=True))
        prev_scale = out_scale

    logit_scale = max(float(np.percentile(np.abs(logits), 99.9)), 1e-3) / 127.0
    level_scale = max(float(np.percentile(np.abs(level), 99.9)), 1.0) / 127.0

    emotion = quantize_layer(net.emotion, prev_scale, 0, logit_scale, 0, -127, apply_relu=False)
    intensity = quantize_layer(net.intensity, prev_scale, 0, level_scale, 0, -127, apply_relu=False)

    return QuantizedNet(
        scaler, input_scale, heart, motion, voice, deep, emotion, intensity,
        logit_scale, level_scale,
    )


def calibrate(model: QuantizedNet, val_set: Dataset) -> None:
    logits, _ = model.forward(model.quantize_inputs(val_set.X))
    scaled = logits.astype(np.float64) * model.logit_scale

    def nll(temperature: float) -> float:
        z = scaled / temperature
        z = z - z.max(axis=1, keepdims=True)
        log_sum_exp = np.log(np.exp(z).sum(axis=1) + 1e-12)
        return float(np.mean(log_sum_exp - z[np.arange(len(z)), val_set.y]))

    model.temperature = float(min(np.linspace(0.5, 5.0, 91), key=nll))

    category, confidence, _ = model.predict(val_set.X)
    hit = category == val_set.y
    best, best_accuracy = 0, -1.0
    for threshold in range(101):
        kept = confidence >= threshold
        if kept.mean() < MIN_KEPT_SHARE:
            break
        acc = float(hit[kept].mean()) if kept.any() else 0.0
        if acc > best_accuracy:
            best, best_accuracy = threshold, acc
        if acc >= TARGET_ACCURACY:
            best = threshold
            break
    model.min_confidence = best


def _fmt_array(arr, fmt_fn):
    return ", ".join(fmt_fn(x) for x in arr)


def print_header_values(model: QuantizedNet) -> None:
    """Generate a complete, ready-to-use emotion_model.h"""
    lines = []
    a = lines.append

    a("// emotion_model.h - Generated automatically by emotion_training.py")
    a("// 8 emotions + stability / intensity / burstiness / wavelength per 15-min window")
    a("#pragma once")
    a("#include <math.h>")
    a("#include <stdint.h>")
    a("")
    a("#define EMOTION_NUM_FEATURES 18")
    a("#define EMOTION_NUM_CLASSES 8")
    a("#define EMOTION_WINDOW_EPOCHS 180")
    a("#define EMOTION_EPOCH_SEC 5")
    a("")
    a("enum EmotionStatus : uint8_t {")
    a("    EMOTION_OK = 0,")
    a("    EMOTION_LOW_CONFIDENCE = 1,")
    a("    EMOTION_HIGH_MOTION = 2,")
    a("    EMOTION_NO_DATA = 3,")
    a("};")
    a("")
    a("static const uint8_t EMOTION_NONE = 255;")
    a('static const char* const EMOTION_NAMES[EMOTION_NUM_CLASSES] = {')
    a('    "calm", "tense", "upbeat", "subdued", "happy", "sad", "angry", "tired"')
    a("};")
    a("")
    a("// Final output for every 15-minute window")
    a("struct EmotionResult {")
    a("    uint8_t status;           // OK / LOW_CONFIDENCE / HIGH_MOTION / NO_DATA")
    a("    uint8_t emotion;          // 0..7 or 255")
    a("    uint8_t confidence_pct;   // 0..100")
    a("    float   stability;        // HR stability (std)")
    a("    float   intensity;        // HR intensity (mean - baseline)")
    a("    float   burstiness;       // HR burstiness (std / |mean|)")
    a("    float   wavelength;       // HR peak interval (seconds)")
    a("};")
    a("")
    a("// ---------- Scaler ----------")
    a("static const float kEmotionMean[18] = {")
    a("    " + _fmt_array(model.scaler.mean, lambda x: f"{x:.9e}f"))
    a("};")
    a("static const float kEmotionStd[18] = {")
    a("    " + _fmt_array(model.scaler.std, lambda x: f"{x:.9e}f"))
    a("};")
    a(f"static const double kEmotionInputScale = {model.input_scale};")
    a(f"static const double kEmotionLogitScale = {model.logit_scale};")
    a(f"static const double kEmotionLevelScale = {model.level_scale};")
    a(f"static const double kEmotionTemperature = {model.temperature};")
    a(f"static const uint8_t kEmotionMinConfidence = {model.min_confidence};")
    a("")

    def dump_layer(name: str, layer: Layer):
        w = layer.weights.ravel().tolist()
        b = layer.bias.tolist()
        a(f"static const int8_t k{name}Weights[{len(w)}] = {{")
        a("    " + ", ".join(str(int(x)) for x in w))
        a("};")
        a(f"static const int32_t k{name}Bias[{len(b)}] = {{")
        a("    " + ", ".join(str(int(x)) for x in b))
        a("};")
        a(f"static const int32_t k{name}Mult = {layer.mult};")
        a(f"static const uint8_t k{name}Shift = {layer.shift};")
        a(f"static const int32_t k{name}InZero = {layer.in_zero};")
        a(f"static const int32_t k{name}OutZero = {layer.out_zero};")
        a(f"static const int32_t k{name}OutMin = {layer.out_min};")
        a("")

    dump_layer("Heart", model.heart)
    dump_layer("Motion", model.motion)
    dump_layer("Voice", model.voice)
    dump_layer("Deep1", model.deep[0])
    dump_layer("Deep2", model.deep[1])
    dump_layer("Deep3", model.deep[2])
    dump_layer("Emotion", model.emotion)
    dump_layer("Intensity", model.intensity)

    a("// ---------- Feature-extraction constants ----------")
    a("static const uint16_t kEmotionMinEpochs[4] = {90, 90, 6, 6};")
    a("static const double kEmotionMeanFloor[4] = {1.0, 10.0, 10.0, 1.0};")
    a("static const double kEmotionBaselinePrior[4] = {70.0, 30.0, 160.0, 30.0};")
    a("static const double kEmotionPeakSigma = 0.5;")
    a("static const double kEmotionActivityGateMg = 150.0;")
    a("static const double kEmotionBaselineAlphaMin = 0.01;")
    a("")

    a("// ---------- Integer dense layer ----------")
    a("static inline bool emotion_present(double x) { return x - x == 0.0; }")
    a("")
    a("static inline void emotion_dense(const int8_t* input, uint8_t n_in, int32_t in_zero,")
    a("                                 const int8_t* weights, const int32_t* bias, int32_t mult,")
    a("                                 uint8_t shift, uint8_t n_out, int32_t out_zero, int32_t out_min,")
    a("                                 int8_t* output) {")
    a("    for (uint8_t k = 0; k < n_out; ++k) {")
    a("        int32_t sum = bias[k];")
    a("        for (uint8_t i = 0; i < n_in; ++i) {")
    a("            sum += (int32_t)(*weights++) * ((int32_t)input[i] - in_zero);")
    a("        }")
    a("        int64_t value = (int64_t)sum * mult;")
    a("        value = (value + ((int64_t)1 << (shift - 1))) >> shift;")
    a("        value += out_zero;")
    a("        if (value < out_min) value = out_min;")
    a("        if (value > 127) value = 127;")
    a("        output[k] = (int8_t)value;")
    a("    }")
    a("}")
    a("")
    a("static inline int8_t emotion_quantize(double value, uint8_t index) {")
    a("    if (!emotion_present(value)) return 0;")
    a("    const double compressed = copysign(log1p(fabs(value)), value);")
    a("    const double z = (compressed - kEmotionMean[index]) / kEmotionStd[index];")
    a("    const double q = floor(z / kEmotionInputScale + 0.5);")
    a("    if (q <= -127.0) return -127;")
    a("    if (q >= 127.0) return 127;")
    a("    return (int8_t)q;")
    a("}")
    a("")
    a("static inline void emotion_forward(const int8_t input[EMOTION_NUM_FEATURES],")
    a("                                   int8_t logits[EMOTION_NUM_CLASSES], int8_t* intensity) {")
    a("    int8_t encoded[20];")
    a("    emotion_dense(input, 4, kHeartInZero, kHeartWeights, kHeartBias, kHeartMult,")
    a("                  kHeartShift, 6, kHeartOutZero, kHeartOutMin, encoded);")
    a("    emotion_dense(input + 4, 4, kMotionInZero, kMotionWeights, kMotionBias, kMotionMult,")
    a("                  kMotionShift, 6, kMotionOutZero, kMotionOutMin, encoded + 6);")
    a("    emotion_dense(input + 8, 10, kVoiceInZero, kVoiceWeights, kVoiceBias, kVoiceMult,")
    a("                  kVoiceShift, 8, kVoiceOutZero, kVoiceOutMin, encoded + 12);")
    a("")
    a("    int8_t deep1[16];")
    a("    emotion_dense(encoded, 20, kDeep1InZero, kDeep1Weights, kDeep1Bias, kDeep1Mult,")
    a("                  kDeep1Shift, 16, kDeep1OutZero, kDeep1OutMin, deep1);")
    a("")
    a("    int8_t deep2[12];")
    a("    emotion_dense(deep1, 16, kDeep2InZero, kDeep2Weights, kDeep2Bias, kDeep2Mult,")
    a("                  kDeep2Shift, 12, kDeep2OutZero, kDeep2OutMin, deep2);")
    a("")
    a("    int8_t deep3[8];")
    a("    emotion_dense(deep2, 12, kDeep3InZero, kDeep3Weights, kDeep3Bias, kDeep3Mult,")
    a("                  kDeep3Shift, 8, kDeep3OutZero, kDeep3OutMin, deep3);")
    a("")
    a("    emotion_dense(deep3, 8, kEmotionInZero, kEmotionWeights, kEmotionBias, kEmotionMult,")
    a("                  kEmotionShift, 8, kEmotionOutZero, kEmotionOutMin, logits);")
    a("    emotion_dense(deep3, 8, kIntensityInZero, kIntensityWeights, kIntensityBias, kIntensityMult,")
    a("                  kIntensityShift, 1, kIntensityOutZero, kIntensityOutMin, intensity);")
    a("}")
    a("")
    a("static inline EmotionResult emotion_decide(const int8_t logits[EMOTION_NUM_CLASSES],")
    a("                                           const double features[EMOTION_NUM_FEATURES]) {")
    a("    EmotionResult result;")
    a("    result.emotion = EMOTION_NONE;")
    a("    result.stability  = (float)features[0];")
    a("    result.intensity  = (float)features[1];")
    a("    result.burstiness = (float)features[2];")
    a("    result.wavelength = (float)features[3];")
    a("")
    a("    uint8_t top = 0;")
    a("    for (uint8_t j = 1; j < EMOTION_NUM_CLASSES; ++j) {")
    a("        if (logits[j] > logits[top]) top = j;")
    a("    }")
    a("    double total = 0.0;")
    a("    for (uint8_t j = 0; j < EMOTION_NUM_CLASSES; ++j) {")
    a("        const double gap = (double)((int16_t)logits[j] - (int16_t)logits[top]);")
    a("        total += exp(gap * kEmotionLogitScale / kEmotionTemperature);")
    a("    }")
    a("    result.confidence_pct = (uint8_t)floor(100.0 / total + 0.5);")
    a("")
    a("    if (result.confidence_pct < kEmotionMinConfidence) {")
    a("        result.status = EMOTION_LOW_CONFIDENCE;")
    a("        return result;")
    a("    }")
    a("    result.status = EMOTION_OK;")
    a("    result.emotion = top;")
    a("    return result;")
    a("}")
    a("")
    a("static inline EmotionResult predict_emotion(const double features[EMOTION_NUM_FEATURES]) {")
    a("    if (!emotion_present(features[0]) || !emotion_present(features[4])) {")
    a("        EmotionResult result;")
    a("        result.status = EMOTION_NO_DATA;")
    a("        result.emotion = EMOTION_NONE;")
    a("        result.confidence_pct = 0;")
    a("        result.stability = result.intensity = result.burstiness = result.wavelength = 0.0f;")
    a("        return result;")
    a("    }")
    a("    int8_t input[EMOTION_NUM_FEATURES];")
    a("    for (uint8_t i = 0; i < EMOTION_NUM_FEATURES; ++i) {")
    a("        input[i] = emotion_quantize(features[i], i);")
    a("    }")
    a("    int8_t logits[EMOTION_NUM_CLASSES];")
    a("    int8_t level;")
    a("    emotion_forward(input, logits, &level);")
    a("    return emotion_decide(logits, features);")
    a("}")
    a("")
    a("// ---------- Sliding 15-minute window ----------")
    a("class EmotionWindow {")
    a("public:")
    a("    EmotionWindow() : filled_(0), mic_epochs_(0) {")
    a("        for (uint8_t c = 0; c < 4; ++c) {")
    a("            baseline_[c] = kEmotionBaselinePrior[c];")
    a("            count_[c] = 0;")
    a("        }")
    a("        for (uint8_t i = 0; i < EMOTION_NUM_FEATURES; ++i) features_[i] = NAN;")
    a("    }")
    a("    bool push(float hr_bpm, float motion_mg, float pitch_hz, float energy_db, bool mic_on,")
    a("              EmotionResult& result) {")
    a("        buffer_[0][filled_] = clean(hr_bpm);")
    a("        buffer_[1][filled_] = clean(motion_mg);")
    a("        buffer_[2][filled_] = mic_on ? clean(pitch_hz) : NAN;")
    a("        buffer_[3][filled_] = mic_on ? clean(energy_db) : NAN;")
    a("        if (mic_on) ++mic_epochs_;")
    a("        if (++filled_ < EMOTION_WINDOW_EPOCHS) return false;")
    a("        finish(result);")
    a("        filled_ = 0;")
    a("        mic_epochs_ = 0;")
    a("        return true;")
    a("    }")
    a("    const double* features() const { return features_; }")
    a("private:")
    a("    static float clean(float x) { return emotion_present(x) ? x : NAN; }")
    a("    void summarize(uint8_t channel, double& mean, double& spread, uint16_t& count) const {")
    a("        double total = 0.0;")
    a("        count = 0;")
    a("        for (uint16_t i = 0; i < EMOTION_WINDOW_EPOCHS; ++i) {")
    a("            const double x = buffer_[channel][i];")
    a("            if (emotion_present(x)) { total += x; ++count; }")
    a("        }")
    a("        const uint16_t divisor = count > 0 ? count : 1;")
    a("        mean = total / divisor;")
    a("        double squares = 0.0;")
    a("        for (uint16_t i = 0; i < EMOTION_WINDOW_EPOCHS; ++i) {")
    a("            const double x = buffer_[channel][i];")
    a("            if (emotion_present(x)) squares += (x - mean) * (x - mean);")
    a("        }")
    a("        spread = sqrt(squares / divisor);")
    a("    }")
    a("    double peakInterval(uint8_t channel, double mean, double spread) const {")
    a("        const double threshold = mean + kEmotionPeakSigma * spread;")
    a("        const float* x = buffer_[channel];")
    a("        uint16_t first = 0, last = 0, peaks = 0;")
    a("        for (uint16_t i = 1; i + 1 < EMOTION_WINDOW_EPOCHS; ++i) {")
    a("            const double before = x[i - 1], here = x[i], after = x[i + 1];")
    a("            if (here > before && here >= after && here > threshold) {")
    a("                if (peaks == 0) first = i;")
    a("                last = i;")
    a("                ++peaks;")
    a("            }")
    a("        }")
    a("        if (peaks < 2) return (double)EMOTION_WINDOW_EPOCHS * EMOTION_EPOCH_SEC;")
    a("        return (double)(last - first) / (peaks - 1) * EMOTION_EPOCH_SEC;")
    a("    }")
    a("    void finish(EmotionResult& result) {")
    a("        double mean[4], spread[4];")
    a("        uint16_t count[4];")
    a("        bool usable[4];")
    a("        for (uint8_t c = 0; c < 4; ++c) {")
    a("            summarize(c, mean[c], spread[c], count[c]);")
    a("            usable[c] = count[c] >= kEmotionMinEpochs[c];")
    a("        }")
    a("        const bool voice_ok = usable[2] && usable[3];")
    a("        for (uint8_t i = 0; i < EMOTION_NUM_FEATURES; ++i) features_[i] = NAN;")
    a("        for (uint8_t c = 0; c < 4; ++c) {")
    a("            if (!usable[c] || (c >= 2 && !voice_ok)) continue;")
    a("            const double magnitude = fabs(mean[c]);")
    a("            const double divisor = magnitude > kEmotionMeanFloor[c] ? magnitude : kEmotionMeanFloor[c];")
    a("            features_[c * 4 + 0] = spread[c];")
    a("            features_[c * 4 + 1] = mean[c] - baseline_[c];")
    a("            features_[c * 4 + 2] = spread[c] / divisor;")
    a("            features_[c * 4 + 3] = peakInterval(c, mean[c], spread[c]);")
    a("        }")
    a("        if (voice_ok) {")
    a("            features_[16] = (double)count[2] / (mic_epochs_ > 0 ? mic_epochs_ : 1);")
    a("            features_[17] = 1.0;")
    a("        } else {")
    a("            features_[17] = 0.0;")
    a("        }")
    a("        if (!usable[0] || !usable[1]) {")
    a("            result.status = EMOTION_NO_DATA;")
    a("            result.emotion = EMOTION_NONE;")
    a("            result.confidence_pct = 0;")
    a("            result.stability = result.intensity = result.burstiness = result.wavelength = 0.0f;")
    a("        } else if (mean[1] >= kEmotionActivityGateMg) {")
    a("            result.status = EMOTION_HIGH_MOTION;")
    a("            result.emotion = EMOTION_NONE;")
    a("            result.confidence_pct = 0;")
    a("            result.stability  = (float)features_[0];")
    a("            result.intensity  = (float)features_[1];")
    a("            result.burstiness = (float)features_[2];")
    a("            result.wavelength = (float)features_[3];")
    a("        } else {")
    a("            result = predict_emotion(features_);")
    a("        }")
    a("        if (usable[1] && mean[1] < kEmotionActivityGateMg) {")
    a("            for (uint8_t c = 0; c < 4; ++c) {")
    a("                if (!usable[c]) continue;")
    a("                ++count_[c];")
    a("                double alpha = 1.0 / (count_[c] + 1.0);")
    a("                if (alpha < kEmotionBaselineAlphaMin) alpha = kEmotionBaselineAlphaMin;")
    a("                baseline_[c] += alpha * (mean[c] - baseline_[c]);")
    a("            }")
    a("        }")
    a("    }")
    a("    float buffer_[4][EMOTION_WINDOW_EPOCHS];")
    a("    uint16_t filled_;")
    a("    uint16_t mic_epochs_;")
    a("    double baseline_[4];")
    a("    uint32_t count_[4];")
    a("    double features_[EMOTION_NUM_FEATURES];")
    a("};")
    a("")

    content = "\n".join(lines)
    with open("emotion_model.h", "w", encoding="utf-8") as f:
        f.write(content)
    print("\n[SUCCESS] Complete file 'emotion_model.h' has been generated!")


def main():
    print(f"Simulating {PEOPLE} people with 8 emotions")
    rng = np.random.default_rng(SEED)
    people = [simulate_person(rng) for _ in range(PEOPLE)]
    print("Extracting 15-minute features")
    data = build_dataset(people)
    order = np.random.default_rng(SEED).permutation(PEOPLE)
    n_test, n_val = PEOPLE // 5, PEOPLE * 3 // 20
    test_ids, val_ids = order[:n_test], order[n_test : n_test + n_val]
    train_ids = order[n_test + n_val :]
    usable = (data.y >= 0) & (data.status == STATUS_OK)
    train_set = data.select(usable & np.isin(data.person, train_ids))
    val_set = data.select(usable & np.isin(data.person, val_ids))
    test_set = data.select(usable & np.isin(data.person, test_ids))
    scaler = Scaler.fit(train_set.X)
    print(f"Training on {len(train_set.y)} windows from {len(train_ids)} people")
    net = train(train_set, val_set, scaler)
    model = quantize(net, train_set, scaler)
    calibrate(model, val_set)
    with torch.no_grad():
        float_logits, _ = net(torch.as_tensor(scaler.inputs(test_set.X)))
    category, confidence, intensity = model.predict(test_set.X)
    hit = category == test_set.y
    kept = confidence >= model.min_confidence
    parameters = sum(p.numel() for p in net.parameters())
    print(f"\nNetwork: {parameters} parameters")
    print(f"Simulated test people: {len(test_ids)}, {len(test_set.y)} windows")
    guessing = np.bincount(test_set.y, minlength=8).max() / max(len(test_set.y), 1)
    print(f"Always guessing the commonest emotion: {guessing:.3f}")
    print(f"Accuracy, float: {(float_logits.argmax(dim=1).numpy() == test_set.y).mean():.3f}")
    print(f"Accuracy, int8: {hit.mean():.3f}")
    print(f"Confidence gate: keeps {kept.mean():.1%} of windows at {hit[kept].mean():.3f} accuracy")
    print(f"Intensity mean error: {np.abs(intensity / 100.0 - test_set.level).mean():.3f}")
    print_header_values(model)
    return people, test_ids, net, model, test_set


if __name__ == "__main__":
    main()
