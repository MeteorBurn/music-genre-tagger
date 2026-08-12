# Copyright 2024 Music Technology Group, Universitat Pompeu Fabra
# Copyright 2026 MAEST 522 community extension contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Derived from the MAEST Hugging Face feature extractor published by MTG-UPF.
"""Exact torchaudio MAEST feature extractor for Hugging Face remote code."""

from __future__ import annotations

from typing import Any

import numpy
import torch
from torch import Tensor
from torchaudio.transforms import MelScale, Spectrogram
from transformers.feature_extraction_utils import BatchFeature, FeatureExtractionMixin


class MAESTFeatureExtractor(FeatureExtractionMixin):
    """Convert raw 16 kHz mono audio into normalized MAEST log-mel frames."""

    model_input_names = ["input_values"]

    def __init__(
        self,
        feature_size: int = 1,
        sampling_rate: int = 16_000,
        n_fft: int = 512,
        hop_length: int = 256,
        num_mel_bins: int = 96,
        max_length: int = 1876,
        mean: float = 2.06755686098554,
        std: float = 1.268292820667291,
        padding_value: float = 0.0,
        return_attention_mask: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        if sampling_rate != 16_000:
            raise ValueError("MAESTFeatureExtractor supports only 16000 Hz")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_mel_bins = num_mel_bins
        self.max_length = max_length
        self.mean = mean
        self.std = std
        self.do_normalize = True
        self.log_compression = "logC"
        self.padding_side = "right"
        self._spectrogram = Spectrogram(
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            power=2,
        )
        self._mel_scale = MelScale(
            n_mels=num_mel_bins,
            sample_rate=sampling_rate,
            n_stft=n_fft // 2 + 1,
            norm="slaney",
            mel_scale="slaney",
        )

    def _extract(self, waveform: Tensor) -> Tensor:
        power = self._spectrogram(waveform)
        mel = self._mel_scale(power)
        logmel = torch.log10(1.0 + mel * 10_000.0)
        normalized = (logmel - self.mean) / (self.std * 2.0)
        return normalized.transpose(0, 1)

    def __call__(
        self,
        raw_speech: Any,
        sampling_rate: int | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        del kwargs
        if sampling_rate is None or sampling_rate != self.sampling_rate:
            raise ValueError(
                f"sampling_rate must be exactly {self.sampling_rate}, got {sampling_rate}"
            )
        if isinstance(raw_speech, (numpy.ndarray, Tensor)):
            array = torch.as_tensor(raw_speech, dtype=torch.float32)
            if array.ndim == 1:
                waveforms = [array]
            elif array.ndim == 2:
                raise ValueError(
                    "MAESTFeatureExtractor accepts mono waveforms; pass a list for batching"
                )
            else:
                raise ValueError("raw audio must be a one-dimensional mono waveform")
        elif isinstance(raw_speech, (list, tuple)):
            waveforms = [torch.as_tensor(item, dtype=torch.float32) for item in raw_speech]
            if any(item.ndim != 1 for item in waveforms):
                raise ValueError("every batched waveform must be one-dimensional mono audio")
        else:
            raise ValueError("unsupported raw audio container")
        features = []
        for waveform in waveforms:
            extracted = self._extract(waveform)
            if extracted.shape[0] >= self.max_length:
                fixed = extracted[: self.max_length]
            else:
                fixed = torch.full(
                    (self.max_length, self.num_mel_bins),
                    float(self.padding_value),
                    dtype=extracted.dtype,
                )
                fixed[: extracted.shape[0]] = extracted
            features.append(fixed.cpu().numpy())
        batch = BatchFeature(
            data={"input_values": numpy.stack(features)},
            tensor_type=return_tensors,
        )
        return batch
