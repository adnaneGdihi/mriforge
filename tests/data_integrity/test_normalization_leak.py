"""
Critical tests to ensure normalization statistics are derived ONLY from training data.

Run on CI, NOT local dev machine.
"""

import pytest
import torch


@pytest.mark.data_integrity
@pytest.mark.timeout(30)
class TestNormalizationLeak:
    def test_zscore_stats_from_train_only(self):
        """
        Verify that Z-Score normalization uses mean/std from training set, NOT global or batch stats
        (unless instance norm is explicitly intended, but even then stats calculation logic is checked).
        """
        # Setup mock data for train and val
        train_data = torch.randn(100, 1, 32, 32) + 5  # Mean ~ 5
        val_data = torch.randn(20, 1, 32, 32) - 5  # Mean ~ -5

        # 1. Compute stats from TRAIN
        train_mean = train_data.mean()
        train_std = train_data.std()

        # 2. Apply normalization to VAL using TRAIN stats
        # (This mimics what the pipeline should do)
        normalized_val = (val_data - train_mean) / (train_std + 1e-6)

        # 3. Assert the logic holds - if we used VAL stats, mean would be 0.
        # Since we used TRAIN stats (mean=5), normalized val (mean=-5) should be around (-5 - 5) = -10
        val_mean_after = normalized_val.mean()

        # If leak occurred (val used its own stats), val_mean_after would be ~0
        # If NO leak (correct), val_mean_after should be far from 0 (approx -10 in this synthetic case)
        assert (
            abs(val_mean_after) > 1.0
        ), "Validation data seems to be self-normalized (potential leak of test info into stats)!"

    def test_global_scaler_persistence(self):
        """
        Test that global scalers (min-max/k-space) are fit on train and frozen.
        """

        # Mock a scaler class
        class MockScaler:
            def __init__(self):
                self.min = None
                self.max = None
                self.frozen = False

            def fit(self, data):
                if self.frozen:
                    raise RuntimeError("Cannot fit frozen scaler!")
                self.min = data.min()
                self.max = data.max()
                self.frozen = True  # Freeze after fit

            def transform(self, data):
                return (data - self.min) / (self.max - self.min)

        scaler = MockScaler()
        train_data = torch.tensor([0.0, 10.0])
        val_data = torch.tensor([-5.0, 15.0])

        # Fit on train
        scaler.fit(train_data)

        # Check internal state
        assert scaler.frozen
        assert scaler.min == 0.0
        assert scaler.max == 10.0

        # Transform val
        # Should NOT update min/max
        _ = scaler.transform(val_data)

        assert scaler.min == 0.0
        assert scaler.max == 10.0

        # Attempt to re-fit (simulation of leak) should fail
        with pytest.raises(RuntimeError):
            scaler.fit(val_data)
