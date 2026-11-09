#!/usr/bin/env python3
"""Sentiant emotion model: data simulation, 15-minute features and network training.
Runs on its own and writes emotion_model.h automatically:
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
PEOPLE = 40
DAYS = 6
HOURS_PER_DAY = 12
EPOCH_SEC = 5
WINDOW_SEC = 900
WINDOW_EPOCHS = WINDOW_SEC // EPOCH_SEC
EMOTIONS = ("calm", "tense", "upbeat", "subdued")
EMOTION_SHARE = (0.45, 0.2, 0.2, 0.15)
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
TARGET_ACCURACY = 0.85
MIN_KEPT_SHARE = 0.2

EFFECTS = np.array(
    [
        [2.0, 1.5, 180.0, 2.0, 1.0, 1.00, 0.040, 0.0, 1.0],
        [16.0, 3.0, 60.0, 20.0, 1.5, 1.09, 0.070, 3.0, 1.1],
        [10.0, 2.5, 110.0, 6.0, 1.8, 1.07, 0.120, 5.0, 1.6],
        [-4.0, 0.8, 240.0, 1.0, 0.6, 0.95, 0.025, -4.0, 0.5],
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
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

    def loss_of(inputs: torch.Tensor, labels: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        logits, predicted = net(inputs)
        emotion_loss = nn.functional.cross_entropy(logits, labels, weight=weight)
        return emotion_loss + 2.0 * nn.functional.mse_loss(predicted, levels)

    best, best_state, stale = math.inf, None, 0
    for _ in range(300):
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
            out = np.maximum(out, self.out_zero)  # ReLU relative to the zero-point
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
        q_weights,
        q_bias,
        mult,
        shift,
        in_zero,
        out_zero,
        out_min,
        apply_relu=apply_relu,
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
    """Post-training quantization with consistent zero-points and explicit ReLU."""
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

    # هر سه شاخه با همان scale و zero-point=0 (بعد از ReLU)
    heart = quantize_layer(net.heart, input_scale, 0, scale0, 0, 0, apply_relu=True)
    motion = quantize_layer(net.motion, input_scale, 0, scale0, 0, 0, apply_relu=True)
    voice = quantize_layer(net.voice, input_scale, 0, scale0, 0, 0, apply_relu=True)

    deep: List[Layer] = []
    prev_scale = scale0
    for i, (layer, state) in enumerate(zip(net.deep, states[1:])):
        out_scale = act_scale(state)
        deep.append(
            quantize_layer(layer, prev_scale, 0, out_scale, 0, 0, apply_relu=True)
        )
        prev_scale = out_scale

    logit_scale = max(float(np.percentile(np.abs(logits), 99.9)), 1e-3) / 127.0
    level_scale = max(float(np.percentile(np.abs(level), 99.9)), 1.0) / 127.0

    emotion = quantize_layer(net.emotion, prev_scale, 0, logit_scale, 0, -127, apply_relu=False)
    intensity = quantize_layer(net.intensity, prev_scale, 0, level_scale, 0, -127, apply_relu=False)

    return QuantizedNet(
        scaler,
        input_scale,
        heart,
        motion,
        voice,
        deep,
        emotion,
        intensity,
        logit_scale,
        level_scale,
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


def print_header_values(model: QuantizedNet) -> None:
    header_content = f"""// emotion_model.h - Generated automatically by emotion_training.py
#pragma once
#include <math.h>
#include <stdint.h>

#define EMOTION_NUM_FEATURES 18
#define EMOTION_NUM_CLASSES 4
#define EMOTION_WINDOW_EPOCHS 180
#define EMOTION_EPOCH_SEC 5

enum EmotionStatus : uint8_t {{
    EMOTION_OK = 0,
    EMOTION_LOW_CONFIDENCE = 1,
    EMOTION_HIGH_MOTION = 2,
    EMOTION_NO_DATA = 3,
}};

static const uint8_t EMOTION_NONE = 255;
static const char* const EMOTION_NAMES[EMOTION_NUM_CLASSES] = {{"calm", "tense", "upbeat", "subdued"}};

struct EmotionResult {{
    uint8_t status;
    uint8_t emotion;
    uint8_t confidence_pct;
    uint8_t intensity_pct;
}};

static const float kEmotionMean[18] = {{
    {', '.join(f'{x:.9e}f' for x in model.scaler.mean.tolist())}
}};

static const float kEmotionStd[18] = {{
    {', '.join(f'{x:.9e}f' for x in model.scaler.std.tolist())}
}};

static const double kEmotionInputScale = {model.input_scale};
static const double kEmotionLogitScale = {model.logit_scale};
static const double kEmotionLevelScale = {model.level_scale};
static const double kEmotionTemperature = {model.temperature};
static const uint8_t kEmotionMinConfidence = {model.min_confidence};

static const int8_t kHeartWeights[{len(model.heart.weights.ravel())}] = {{
    {', '.join(map(str, model.heart.weights.ravel().tolist()))}
}};
static const int32_t kHeartBias[{len(model.heart.bias)}] = {{
    {', '.join(map(str, model.heart.bias.tolist()))}
}};

static const int8_t kMotionWeights[{len(model.motion.weights.ravel())}] = {{
    {', '.join(map(str, model.motion.weights.ravel().tolist()))}
}};
static const int32_t kMotionBias[{len(model.motion.bias)}] = {{
    {', '.join(map(str, model.motion.bias.tolist()))}
}};

static const int8_t kVoiceWeights[{len(model.voice.weights.ravel())}] = {{
    {', '.join(map(str, model.voice.weights.ravel().tolist()))}
}};
static const int32_t kVoiceBias[{len(model.voice.bias)}] = {{
    {', '.join(map(str, model.voice.bias.tolist()))}
}};

static const int8_t kDeep1Weights[{len(model.deep[0].weights.ravel())}] = {{
    {', '.join(map(str, model.deep[0].weights.ravel().tolist()))}
}};
static const int32_t kDeep1Bias[{len(model.deep[0].bias)}] = {{
    {', '.join(map(str, model.deep[0].bias.tolist()))}
}};

static const int8_t kDeep2Weights[{len(model.deep[1].weights.ravel())}] = {{
    {', '.join(map(str, model.deep[1].weights.ravel().tolist()))}
}};
static const int32_t kDeep2Bias[{len(model.deep[1].bias)}] = {{
    {', '.join(map(str, model.deep[1].bias.tolist()))}
}};

static const int8_t kDeep3Weights[{len(model.deep[2].weights.ravel())}] = {{
    {', '.join(map(str, model.deep[2].weights.ravel().tolist()))}
}};
static const int32_t kDeep3Bias[{len(model.deep[2].bias)}] = {{
    {', '.join(map(str, model.deep[2].bias.tolist()))}
}};

static const int8_t kEmotionWeights[{len(model.emotion.weights.ravel())}] = {{
    {', '.join(map(str, model.emotion.weights.ravel().tolist()))}
}};
static const int32_t kEmotionBias[{len(model.emotion.bias)}] = {{
    {', '.join(map(str, model.emotion.bias.tolist()))}
}};

static const int8_t kIntensityWeights[{len(model.intensity.weights.ravel())}] = {{
    {', '.join(map(str, model.intensity.weights.ravel().tolist()))}
}};
static const int32_t kIntensityBias[{len(model.intensity.bias)}] = {{
    {', '.join(map(str, model.intensity.bias.tolist()))}
}};
"""
    with open("emotion_model.h", "w", encoding="utf-8") as f:
        f.write(header_content)
    print("\n[SUCCESS] File 'emotion_model.h' has been generated automatically!")


def main() -> Tuple[List[Dict[str, np.ndarray]], np.ndarray, FusionNet, QuantizedNet, Dataset]:
    print(f"Simulating {PEOPLE} people")
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
    guessing = np.bincount(test_set.y).max() / len(test_set.y)
    print(f"Always guessing the commonest emotion: {guessing:.3f}")
    print(f"Accuracy, float: {(float_logits.argmax(dim=1).numpy() == test_set.y).mean():.3f}")
    print(f"Accuracy, int8: {hit.mean():.3f}")
    print(f"Confidence gate: keeps {kept.mean():.1%} of windows at {hit[kept].mean():.3f} accuracy")
    print(f"Intensity mean error: {np.abs(intensity / 100.0 - test_set.level).mean():.3f}")
    print_header_values(model)
    return people, test_ids, net, model, test_set


if __name__ == "__main__":
    main()