"""Deterministic subsets of unique hateful messages, sampled per dataset."""

import pandas as pd


def sample_messages(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if limit < 0:
        raise ValueError("LIMIT cannot be negative.")
    if frame["hate_speech"].isna().any():
        raise ValueError("Dataset contains missing hateful messages.")
    frame = frame.sort_values("id", kind="stable")
    keys = frame["hate_speech"].astype(str).str.strip()
    if keys.eq("").any():
        raise ValueError("Dataset contains empty hateful messages.")
    # MultiCONAN may contain multiple reference responses to the same message.
    frame = frame.loc[~keys.duplicated()].copy()
    if limit > 0:
        frame = frame.sample(n=min(limit, len(frame)), random_state=seed)
    return frame.reset_index(drop=True)
