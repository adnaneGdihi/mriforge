#!/usr/bin/env python
"""PAPI Performance Profiling Tool for Deep Learning Training - Simplified Version

This tool profiles CPU performance counters during neural network training using PAPI.
It automatically cycles through different sets of hardware performance counters,
collecting comprehensive performance metrics across multiple training runs.

Key Features:
- Automatic event set rotation (one event set per training run)
- Comprehensive CPU performance counter coverage
- Simplified DataFrame with only: region, event_name, execution_time_sec, event_value
- Export to Excel and JSON formats
- Progress tracking and status reporting

Usage:
    profiler = TrainingProfiler()

    # Profile training epochs until all event sets are covered
    while profiler.has_remaining_event_sets():
        profiler.start("training_epoch")
        # Your training code here
        profiler.stop("training_epoch")

    profiler.save_metrics("training_session")
"""

import logging
import os
import time
from typing import Any, cast

import pandas as pd

logger = logging.getLogger(__name__)

# Try to import PAPI - gracefully handle if not available
try:
    from pypapi import papi_low as papi

    # from pypapi.events import *  # type: ignore

    PAPI_AVAILABLE = True
except ImportError:
    # print(f"Warning: PAPI not available ({e}). Profiling will be disabled.")
    # Create dummy constants for fallback
    papi = None
    PAPI_AVAILABLE = False
    # Define dummy event constants
    PAPI_BR_CN = 0
    PAPI_BR_INS = 1
    PAPI_BR_MSP = 2
    PAPI_BR_NTK = 3
    PAPI_BR_PRC = 4
    PAPI_BR_TKN = 5
    PAPI_BR_UCN = 6
    PAPI_CA_CLN = 7
    PAPI_CA_INV = 8
    PAPI_CA_ITV = 9
    PAPI_CA_SHR = 10
    PAPI_CA_SNP = 11
    PAPI_DP_OPS = 12
    PAPI_FUL_CCY = 13
    PAPI_FUL_ICY = 14
    PAPI_L1_DCM = 15
    PAPI_L1_ICM = 16
    PAPI_L1_LDM = 17
    PAPI_L1_STM = 18
    PAPI_L1_TCM = 19
    PAPI_L2_DCA = 20
    PAPI_L2_DCM = 21
    PAPI_L2_DCR = 22
    PAPI_L2_DCW = 23
    PAPI_L2_ICA = 24
    PAPI_L2_ICH = 25
    PAPI_L2_ICM = 26
    PAPI_L2_ICR = 27
    PAPI_L2_LDM = 28
    PAPI_L2_STM = 29
    PAPI_L2_TCA = 30
    PAPI_L2_TCM = 31
    PAPI_L2_TCR = 32
    PAPI_L2_TCW = 33
    PAPI_L3_DCA = 34
    PAPI_L3_DCR = 35
    PAPI_L3_DCW = 36
    PAPI_L3_ICA = 37
    PAPI_L3_ICR = 38
    PAPI_L3_LDM = 39
    PAPI_L3_TCA = 40
    PAPI_L3_TCM = 41
    PAPI_L3_TCR = 42
    PAPI_L3_TCW = 43
    PAPI_LD_INS = 44
    PAPI_LST_INS = 45
    PAPI_MEM_WCY = 46
    PAPI_PRF_DM = 47
    PAPI_REF_CYC = 48
    PAPI_RES_STL = 49
    PAPI_SP_OPS = 50
    PAPI_SR_INS = 51
    PAPI_STL_CCY = 52
    PAPI_STL_ICY = 53
    PAPI_TLB_DM = 54
    PAPI_TLB_IM = 55
    PAPI_TOT_CYC = 56
    PAPI_TOT_INS = 57
    PAPI_VEC_DP = 58
    PAPI_VEC_SP = 59


class TrainingProfiler:
    """A performance profiler that cycles through PAPI event sets to comprehensively
    monitor CPU performance during neural network training.

    This profiler automatically manages multiple event sets, ensuring that only
    one event set is active at a time, and progresses through all available
    event sets across multiple training sessions.    DataFrame Structure (Enhanced Timing):
    - region: Name of the profiled region (e.g., "training_epoch")
    - event_name: Name of the PAPI event (e.g., "PAPI_L1_DCM")
    - wall_clock_time_sec: Wall-clock execution time in seconds
    - papi_time_sec: PAPI high-resolution timestamp in seconds (from microseconds)
    - event_value: Measured counter value for the event
    """

    def __init__(
        self,
        output_dir: str = "profiling_logs",
        max_epochs: int = 30,
        batch_based_cycling: bool = True,
    ) -> None:
        """Initialize the profiler with PAPI event sets.

        Args:
            output_dir (str): Directory to save profiling results
            max_epochs (int): Maximum number of epochs to profile (default: 30)
            batch_based_cycling (bool): Whether to cycle event sets every batch (default: True)

        """
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.batch_based_cycling = batch_based_cycling
        self.current_epoch = 0
        self.current_batch = 0
        os.makedirs(self.output_dir, exist_ok=True)

        # Define comprehensive set of PAPI events to monitor
        self.papi_events = [
            # Cache miss events
            PAPI_L1_DCM,
            PAPI_L1_ICM,
            PAPI_L2_DCM,
            PAPI_L2_ICM,
            PAPI_L1_TCM,
            PAPI_L2_TCM,
            PAPI_L3_TCM,
            PAPI_L3_LDM,
            PAPI_L1_LDM,
            PAPI_L1_STM,
            PAPI_L2_LDM,
            PAPI_L2_STM,
            PAPI_MEM_WCY,
            # Cache coherence events
            PAPI_CA_SNP,
            PAPI_CA_SHR,
            PAPI_CA_CLN,
            PAPI_CA_INV,
            PAPI_CA_ITV,
            # TLB and prefetch events
            PAPI_TLB_DM,
            PAPI_TLB_IM,
            PAPI_PRF_DM,
            # Pipeline stall events
            PAPI_STL_ICY,
            PAPI_FUL_ICY,
            PAPI_STL_CCY,
            PAPI_FUL_CCY,
            PAPI_RES_STL,
            # Branch prediction events
            PAPI_BR_UCN,
            PAPI_BR_CN,
            PAPI_BR_TKN,
            PAPI_BR_NTK,
            PAPI_BR_MSP,
            PAPI_BR_PRC,
            # Instruction and cycle counts
            PAPI_TOT_INS,
            PAPI_LD_INS,
            PAPI_SR_INS,
            PAPI_BR_INS,
            PAPI_TOT_CYC,
            PAPI_LST_INS,
            PAPI_REF_CYC,
            # Detailed cache access patterns
            PAPI_L2_DCA,
            PAPI_L3_DCA,
            PAPI_L2_DCR,
            PAPI_L3_DCR,
            PAPI_L2_DCW,
            PAPI_L3_DCW,
            PAPI_L2_ICH,
            PAPI_L2_ICA,
            PAPI_L3_ICA,
            PAPI_L2_ICR,
            PAPI_L3_ICR,
            PAPI_L2_TCA,
            PAPI_L3_TCA,
            PAPI_L2_TCR,
            PAPI_L3_TCR,
            PAPI_L2_TCW,
            PAPI_L3_TCW,
            # Floating point operations
            PAPI_SP_OPS,
            PAPI_DP_OPS,
            PAPI_VEC_SP,
            PAPI_VEC_DP,
        ]

        # Profiling state
        self.papi_initialized: bool = False
        self.hw_counters: int = 0
        self.event_sets: list[Any] = []  # list of PAPI event set handles
        self.event_set_names: list[list[str]] = []  # list of event name lists for each set
        self.current_event_set_idx: int = 0  # Index of current event set to use
        self.active_region: str | None = None  # Currently active profiling region

        # Timing and data collection
        self.start_time: float | None = None
        self.papi_start_time: int | None = None

        # Initialize DataFrame columns - will be updated after PAPI
        # initialization
        self.metric_df = pd.DataFrame()

        # Initialize PAPI and create event sets
        self._initialize_papi()

        # Initialize DataFrame with proper columns after PAPI initialization
        if self.papi_initialized and PAPI_AVAILABLE:
            columns = [
                "region",
                "epoch",
                "batch",
                "wall_clock_time_sec",
                "papi_time_sec",
            ] + [papi.event_code_to_name(event) for event in self.papi_events]
        else:
            columns = ["region", "epoch", "batch", "wall_clock_time_sec"]

        self.metric_df = pd.DataFrame(columns=columns)

        # Print conflict summary if we encountered issues
        self._print_initialization_summary()

    def _print_initialization_summary(self) -> None:
        """Print a summary of PAPI initialization and any issues encountered."""
        if not self.papi_initialized:
            # print("\n❌ PAPI initialization failed. Hardware profiling is disabled.")
            return

        # print(f"\n[OK] PAPI successfully initialized")
        # print(f"   Hardware counters available: {self.hw_counters}")
        # print(f"   Event sets created: {len(self.event_sets)}")

        if len(self.event_sets) == 0:
            # print("   ⚠️  No event sets were created - check event compatibility")
            pass
        else:
            # print(f"   Total events to profile: "
            #        f"{sum(len(names) for names in self.event_set_names)}")

            # Show event set distribution
            set_sizes = [len(names) for names in self.event_set_names]
            # print(f"   Event set sizes: {set_sizes}")

            if len(set(set_sizes)) > 1:
                # print("   ℹ️  Variable set sizes indicate some events had conflicts")
                pass

    def _initialize_papi(self) -> None:
        """Initialize PAPI library and create event sets from available events."""
        # Check if PAPI is available
        if not PAPI_AVAILABLE or papi is None:
            logger.warning("PAPI profiling not available - initializing fallback mode")
            self.papi_initialized = False
            self.hw_counters = 0
            self.event_sets = []
            self.current_event_set_index = 0
            self.profiling_data: list[dict[str, Any]] = []
            return

        # print(f"Initializing PAPI with {len(self.papi_events)} performance events...")

        try:
            # Initialize PAPI library
            try:
                ret = papi.library_init(papi.PAPI_VER_CURRENT)
            except TypeError:
                ret = papi.library_init()

            if ret != papi.PAPI_VER_CURRENT:
                # print(f"Warning: PAPI version mismatch. Expected {papi.PAPI_VER_CURRENT}, got {ret}")
                pass

            # Get available hardware counters
            try:
                self.hw_counters = papi.num_hwctrs()
            except AttributeError:
                self.hw_counters = papi.num_cmp_hwctrs(0)

            # print(f"Available hardware counters: {self.hw_counters}")

            # Test which events are supported
            supported_events = self._test_event_support()

            if not supported_events:
                # print("No supported PAPI events found!")
                return

            # Create event sets that fit within hardware counter limits
            self._create_event_sets(supported_events)

            # Split event sets into individual events
            self._split_event_sets()

            self.papi_initialized = len(self.event_sets) > 0
            # print(f"Successfully initialized {len(self.event_sets)} individual event sets")

        except Exception:
            # print(f"Error initializing PAPI: {e}")
            self.papi_initialized = False

    def _split_event_sets(self) -> None:
        """Split each event set into individual events."""
        # print("Splitting event sets into individual events...")

        new_event_sets = []
        new_event_set_names = []

        for _event_set, event_names in zip(self.event_sets, self.event_set_names, strict=False):
            for event_name in event_names:
                try:
                    individual_set = papi.create_eventset()
                    time.sleep(0.1)  # Add 100ms sleep after create_eventset
                    papi.add_event(individual_set, papi.event_name_to_code(event_name))

                    new_event_sets.append(individual_set)
                    new_event_set_names.append([event_name])

                    # print(f"Created individual event set for event: {event_name}")

                except Exception:
                    # print(f"Failed to create individual event set for {event_name}: {e}")

                    # Clean up failed event set
                    try:
                        papi.cleanup_eventset(individual_set)
                        papi.destroy_eventset(individual_set)
                    except Exception as _exc:
                        logger.debug("Suppressed exception: %s", _exc)

        self.event_sets = new_event_sets
        self.event_set_names = new_event_set_names
        # print(f"Successfully split into {len(self.event_sets)} individual event sets.")

    def _test_event_support(self) -> list[Any]:
        """Test which PAPI events are supported on this system."""
        supported_events = []
        unsupported_count = 0

        for event in self.papi_events:
            try:
                test_set = papi.create_eventset()
                time.sleep(0.1)  # Add 100ms sleep after create_eventset
                papi.add_event(test_set, event)
                supported_events.append(event)

                # Clean up test eventset
                try:
                    papi.cleanup_eventset(test_set)
                    papi.destroy_eventset(test_set)
                except Exception as _exc:
                    logger.debug("Suppressed exception: %s", _exc)

            except Exception:
                unsupported_count += 1
        # print(f"Found {len(supported_events)} supported events, {unsupported_count} unsupported")
        return supported_events

    def _create_event_sets(self, supported_events: list[Any]) -> None:
        """Create PAPI event sets where each set contains only one event."""
        # print(f"Creating individual event sets for {len(supported_events)} events...")

        for event in supported_events:
            try:
                individual_set = papi.create_eventset()
                time.sleep(0.1)  # Add 100ms sleep after create_eventset
                papi.add_event(individual_set, event)

                event_name = papi.event_code_to_name(event)
                self.event_sets.append(individual_set)
                self.event_set_names.append([event_name])

                # print(f"Created individual event set for event: {event_name}")

            except Exception:
                event_name = (
                    papi.event_code_to_name(event)
                    if hasattr(papi, "event_code_to_name")
                    else str(event)
                )
                # print(f"Failed to create individual event set for {event_name}: {e}")

                # Clean up failed event set
                try:
                    papi.cleanup_eventset(individual_set)
                    papi.destroy_eventset(individual_set)
                except Exception as _exc:
                    logger.debug("Suppressed exception: %s", _exc)

        # print(f"Successfully created {len(self.event_sets)} individual event sets.")

    def reset_event_sets(self) -> None:
        """Reset the event set index to start profiling from the beginning."""
        self.current_event_set_idx = 0
        self.current_epoch = 0
        self.current_batch = 0
        # print(f"Reset event set index. Ready to profile {len(self.event_sets)} event sets for {self.max_epochs} epochs.")

    def has_remaining_event_sets(self) -> bool:
        """Check if there are more event sets to profile or if we should continue for 30 epochs."""
        if self.batch_based_cycling:
            # Continue for exactly 30 epochs, cycling through event sets
            return self.current_epoch < self.max_epochs
        # Original behavior: stop when all event sets are profiled
        return self.current_event_set_idx < len(self.event_sets)

    def get_progress_info(self) -> dict[str, Any]:
        """Get current profiling progress information."""
        return {
            "current_event_set": self.current_event_set_idx,
            "total_event_sets": len(self.event_sets),
            "remaining_event_sets": len(self.event_sets) - self.current_event_set_idx,
            "progress_percentage": (
                (self.current_event_set_idx / len(self.event_sets)) * 100 if self.event_sets else 0
            ),
        }

    def get_epoch_progress(self) -> dict[str, Any]:
        """Get current epoch progress information."""
        return {
            "current_epoch": self.current_epoch,
            "max_epochs": self.max_epochs,
            "current_batch": self.current_batch,
            "progress_percentage": (
                (self.current_epoch / self.max_epochs) * 100 if self.max_epochs > 0 else 0
            ),
        }

    def get_event_set_info(self) -> dict[str, Any]:
        """Get information about available event sets."""
        info: dict[str, Any] = {
            "total_event_sets": len(self.event_sets),
            "current_event_set_idx": self.current_event_set_idx,
            "remaining_event_sets": len(self.event_sets) - self.current_event_set_idx,
            "event_sets": [],
        }

        event_sets_list = cast("list[dict[str, Any]]", info["event_sets"])
        for i, event_names in enumerate(self.event_set_names):
            event_sets_list.append(
                {
                    "index": i,
                    "event_count": len(event_names),
                    "event_names": event_names,
                    "completed": i < self.current_event_set_idx,
                },
            )

        return info

    def print_event_set_status(self) -> None:
        """Print the current status of event sets."""
        info = self.get_event_set_info()
        # print(f"\nEvent Set Status:")
        # print(f"Total event sets: {info['total_event_sets']}")
        # print(f"Current event set index: {info['current_event_set_idx']}")
        # print(f"Remaining event sets: {info['remaining_event_sets']}")
        # print(f"Event sets:")

        for _es_info in info["event_sets"]:
            # print(f"  [{es_info['index']}] "
            #       f"{'COMPLETED' if es_info['completed'] else 'PENDING'}: "
            #       f"{es_info['event_count']} events - {es_info['event_names']}"
            # f"{' <- CURRENT' if es_info['index'] == self.current_event_set_idx else
            # ''}")
            pass

        # Function continues below

    def start(self, region_name: str, event_set: Any = None) -> bool:
        """Start profiling for the specified event set.

        Args:
            region_name (str): Name of the region to profile
            event_set (object, optional): Event set handle to use.
                                         If None, uses the current event set index.

        """
        # Only allow one region to be profiled at a time
        if self.active_region is not None:
            # print(f"Error: Region '{self.active_region}' is already being profiled. Stop it before starting '{region_name}'.")
            return False

        if not self.papi_initialized or not self.event_sets:
            # print("PAPI not properly initialized. Skipping hardware counter profiling.")
            return False

        # Determine which event set to use
        if event_set is not None:
            # Validate the provided event set
            if event_set not in self.event_sets:
                # print("Error: Provided event set is not valid or not initialized.")
                return False
            target_event_set = event_set
            target_event_set_idx = self.event_sets.index(event_set)
            # print(f"Using specified event set at index {target_event_set_idx}")
        else:
            # Use the current event set index
            if not self.has_remaining_event_sets():
                # print(f"All {len(self.event_sets)} event sets have been profiled. Call reset_event_sets() to start over.")
                return False
            target_event_set_idx = self.current_event_set_idx
            target_event_set = self.event_sets[target_event_set_idx]
            # print(f"Using current event set {self.current_event_set_idx}")

        # Set up profiling state with dual timing
        self.start_time = time.time()
        self.active_region = region_name

        # Update current event set index to match what we're using
        self.current_event_set_idx = target_event_set_idx

        try:
            # Start PAPI counters and capture PAPI timestamp (microseconds)
            papi.start(target_event_set)
            self.papi_start_time = papi.get_real_nsec()

            # print(f"Started profiling event set {self.current_event_set_idx} "
            #       f"for region '{region_name}'")

            # Ensure the region exists in the DataFrame with epoch and batch
            # info
            if region_name not in self.metric_df["region"].values:
                new_row = pd.DataFrame(
                    {
                        "region": [region_name],
                        "epoch": [self.current_epoch],
                        "batch": [self.current_batch],
                        **{
                            event_name: None
                            for event_name in self.metric_df.columns
                            if event_name not in ["region", "epoch", "batch"]
                        },
                    },
                )
                self.metric_df = pd.concat([self.metric_df, new_row], ignore_index=True)

            return True

        except Exception:
            # print(f"Error starting PAPI counters for event set {self.current_event_set_idx}: {e}")
            self.active_region = None
            return False

    def stop(self, region_name: str, event_set: Any = None) -> bool:
        """Stop profiling for the specified event set and advance to next event set."""
        if not self.papi_initialized or not self.event_sets:
            # print("PAPI not properly initialized. Skipping hardware counter profiling.")
            return False

        if self.active_region != region_name:
            # print(f"Warning: Region '{region_name}' is not currently being profiled.")
            return False

        # Determine which event set to use
        if event_set is not None:
            # Validate the provided event set
            if event_set not in self.event_sets:
                # print("Error: Provided event set is not valid or not initialized.")
                return False
            target_event_set = event_set
            target_event_set_idx = self.event_sets.index(event_set)
        else:
            target_event_set_idx = self.current_event_set_idx
            target_event_set = self.event_sets[target_event_set_idx]

        # Capture end times for both timing methods
        end_time = time.time()
        papi_end_time = papi.get_real_nsec()

        # Calculate timing durations
        wall_clock_time = end_time - (self.start_time or end_time)  # Fallback if start_time is None
        # Ensure proper handling of PAPI timing values
        try:
            # Calculate PAPI time in seconds
            papi_time = (papi_end_time - (self.papi_start_time or papi_end_time)) / 1e9
        except (ValueError, AttributeError, TypeError):
            # print(f"Error calculating PAPI time: {e}")
            papi_time = 0

        event_names = self.event_set_names[
            target_event_set_idx
        ]  # Ensure event_names is defined before use

        try:
            # Stop profiling and get results
            raw_values = papi.stop(target_event_set)
            papi.reset(target_event_set)

            if raw_values and len(raw_values) == len(event_names):
                # print(f"Stopped profiling event set {target_event_set_idx} for region '{region_name}': {len(raw_values)} values")
                # print(f"  Wall clock time: {wall_clock_time:.6f}s, PAPI time: {papi_time:.6f}s")

                # Update DataFrame with measured values and timing
                current_row_mask = (
                    (self.metric_df["region"] == region_name)
                    & (self.metric_df["epoch"] == self.current_epoch)
                    & (self.metric_df["batch"] == self.current_batch)
                )

                if not current_row_mask.any():
                    # Add new row for this region/epoch/batch combination
                    new_row = pd.DataFrame(
                        {
                            "region": [region_name],
                            "epoch": [self.current_epoch],
                            "batch": [self.current_batch],
                            "wall_clock_time_sec": [wall_clock_time],
                            "papi_time_sec": [papi_time],
                            **{
                                event_name: ([event_value] if event_name in event_names else [None])
                                for event_name, event_value in zip(
                                    event_names,
                                    raw_values,
                                    strict=False,
                                )
                            },
                            **{
                                col: [None]
                                for col in self.metric_df.columns
                                if col
                                not in [
                                    "region",
                                    "epoch",
                                    "batch",
                                    "wall_clock_time_sec",
                                    "papi_time_sec",
                                ]
                                + event_names
                            },
                        },
                    )
                    self.metric_df = pd.concat(
                        [self.metric_df, new_row],
                        ignore_index=True,
                    )
                else:
                    # Update existing row
                    for event_name, event_value in zip(event_names, raw_values, strict=False):
                        self.metric_df.loc[current_row_mask, event_name] = event_value
                    self.metric_df.loc[current_row_mask, "wall_clock_time_sec"] = wall_clock_time
                    self.metric_df.loc[current_row_mask, "papi_time_sec"] = papi_time

            else:
                # print(f"Warning: Expected {len(event_names)} values, got {len(raw_values) if raw_values else 0}")
                self._update_timing_fallback(
                    region_name,
                    event_names,
                    wall_clock_time,
                    papi_time,
                )

        except Exception:
            # print(f"Error stopping PAPI counters: {e}")
            self._update_timing_fallback(
                region_name,
                event_names,
                wall_clock_time,
                papi_time,
            )

        # Advance to next event set
        if self.batch_based_cycling:
            # Cycle through event sets every batch
            self.current_batch += 1
            self.current_event_set_idx = (self.current_event_set_idx + 1) % len(
                self.event_sets,
            )
        else:
            # Original behavior: advance linearly
            self.current_event_set_idx += 1

        self.active_region = None

        # Print progress
        progress = self.get_progress_info()
        if self.batch_based_cycling:
            logger.info(
                "Completed event set %s for region '%s' (Epoch %s/%s, Batch %s, Event Set %s/%s)",
                target_event_set_idx,
                region_name,
                self.current_epoch + 1,
                self.max_epochs,
                self.current_batch,
                target_event_set_idx + 1,
                len(self.event_sets),
            )
        else:
            logger.info(
                "Completed event set %s for region '%s' (%s/%s, %.1f%% complete)",
                self.current_event_set_idx - 1,
                region_name,
                progress["current_event_set"],
                progress["total_event_sets"],
                progress["progress_percentage"],
            )

        if not self.has_remaining_event_sets():
            if self.batch_based_cycling:
                logger.info("Completed profiling for %s epochs!", self.max_epochs)
            else:
                logger.info("All %s event sets have been profiled!", len(self.event_sets))

        return True

    def _update_timing_fallback(
        self,
        region_name: str,
        event_names: list[str],
        wall_clock_time: float,
        papi_time: float,
    ) -> None:
        """Update timing information when event values are unavailable."""
        for event_name in event_names:
            # Find the most recent row for this region-event combination that hasn't
            # been updated
            condition = (
                (self.metric_df["region"] == region_name)
                & (self.metric_df["event_name"] == event_name)
                & (self.metric_df["wall_clock_time_sec"].isna())
            )
            matching_rows = self.metric_df.loc[condition]

            if not matching_rows.empty:
                last_idx = matching_rows.index[-1]
                self.metric_df.loc[last_idx, "wall_clock_time_sec"] = wall_clock_time
                self.metric_df.loc[last_idx, "papi_time_sec"] = papi_time
                self.metric_df.loc[last_idx, "event_value"] = 0  # Set to 0 when unavailable
                # print(f"  {event_name}: Fallback (value unavailable)")
            else:
                # Create new row if no matching row found
                new_row = pd.DataFrame(
                    {
                        "region": [region_name],
                        "event_name": [event_name],
                        "wall_clock_time_sec": [wall_clock_time],
                        "papi_time_sec": [papi_time],
                        "event_value": [0],
                    },
                )
                self.metric_df = pd.concat([self.metric_df, new_row], ignore_index=True)

    def save_metrics(self, tag: str) -> None:
        """Save the metrics dataframe to Excel file only, updating existing entries with new event values."""
        excel_path = os.path.join(self.output_dir, f"metrics_{tag}.xlsx")

        try:
            # Handle Excel file - merge with existing data, updating matching
            # entries
            if os.path.exists(excel_path):
                try:
                    existing_df = pd.read_excel(excel_path)

                    # Create a combined dataframe by updating existing entries with new
                    # event values
                    combined_df = existing_df.copy()

                    # Process each row in the new data
                    for _, new_row in self.metric_df.iterrows():
                        region_name = new_row.get("region", "")
                        epoch_num = new_row.get("epoch", 0)
                        batch_num = new_row.get("batch", 0)

                        # Find matching row in existing data
                        match_condition = (
                            (combined_df["region"] == region_name)
                            & (combined_df["epoch"] == epoch_num)
                            & (combined_df["batch"] == batch_num)
                        )

                        if match_condition.any():
                            # Update existing row with new event values
                            matching_idx = combined_df[match_condition].index[0]
                            for col in new_row.index:
                                if col not in ["region", "epoch", "batch"]:
                                    value = new_row[col]
                                    # Check if value is not NaN/None and update
                                    if value is not None and str(value).lower() != "nan":
                                        combined_df.loc[matching_idx, col] = value
                        else:
                            # Add new row if no match found
                            combined_df = pd.concat(
                                [combined_df, new_row.to_frame().T],
                                ignore_index=True,
                            )

                    combined_df.to_excel(excel_path, index=False)
                    # print(f"Updated metrics in Excel file: {excel_path}")

                except Exception:
                    # print(f"Error updating existing metrics file: {inner_e}")
                    # Fall back to saving as new file
                    self.metric_df.to_excel(excel_path, index=False)
                    # print(f"Metrics saved to Excel file: {excel_path}")
            else:
                # Save as new Excel file
                self.metric_df.to_excel(excel_path, index=False)
                # print(f"Metrics saved to Excel file: {excel_path}")

        except Exception:
            # print(f"Error saving metrics: {e}")
            pass

    def __del__(self) -> None:
        """__del__."""
        if (
            hasattr(self, "papi_initialized")
            and self.papi_initialized
            and hasattr(self, "event_sets")
        ):
            for _idx, event_set in enumerate(self.event_sets):
                try:
                    # Try to stop counting if it's still running
                    try:
                        papi.stop(event_set)
                    except Exception as stop_e:
                        # Expected if not running, ignore ENOTRUN errors
                        if "ENOTRUN" not in str(stop_e) and "not running" not in str(
                            stop_e,
                        ):
                            # print(f"Warning: Error stopping eventset {idx}: {stop_e}")
                            pass

                    # Clean up the eventset
                    papi.cleanup_eventset(event_set)
                    papi.destroy_eventset(event_set)
                except Exception:
                    # print(f"Error cleaning up event set {idx}: {e}")
                    pass

    def get_timing_analysis(self) -> dict[str, Any]:
        """Analyze timing data to compare wall clock vs PAPI timing accuracy.

        Returns:
            dict: Analysis of timing differences and statistics

        """
        if self.metric_df.empty:
            return {"error": "No timing data available"}

        # Filter out rows without timing data
        valid_data = self.metric_df.dropna(
            subset=["wall_clock_time_sec", "papi_time_sec"],
        )

        if valid_data.empty:
            return {"error": "No valid timing data found"}

        # Calculate timing differences
        timing_diff = valid_data["wall_clock_time_sec"] - valid_data["papi_time_sec"]

        # Convert Timedelta to seconds before calculating statistics
        timing_diff = pd.to_timedelta(timing_diff).dt.total_seconds()
        analysis: dict[str, Any] = {
            "total_measurements": len(valid_data),
            "wall_clock_stats": {
                "mean": float(valid_data["wall_clock_time_sec"].mean()),
                "std": float(valid_data["wall_clock_time_sec"].std()),
                "min": float(valid_data["wall_clock_time_sec"].min()),
                "max": float(valid_data["wall_clock_time_sec"].max()),
            },
            "papi_stats": {
                "mean": float(valid_data["papi_time_sec"].mean()),
                "std": float(valid_data["papi_time_sec"].std()),
                "min": float(valid_data["papi_time_sec"].min()),
                "max": float(valid_data["papi_time_sec"].max()),
            },
            "timing_difference_stats": {
                "mean_diff": float(timing_diff.mean()),
                "std_diff": float(timing_diff.std()),
                "max_abs_diff": float(timing_diff.abs().max()),
                "correlation": float(
                    valid_data["wall_clock_time_sec"].corr(valid_data["papi_time_sec"]),
                ),
            },
        }

        # Add per-region analysis
        analysis["by_region"] = {}
        for region in valid_data["region"].unique():
            region_data = valid_data[valid_data["region"] == region]
            region_diff = region_data["wall_clock_time_sec"] - region_data["papi_time_sec"]

            analysis["by_region"][region] = {
                "measurements": len(region_data),
                "wall_clock_mean": float(region_data["wall_clock_time_sec"].mean()),
                "papi_mean": float(region_data["papi_time_sec"].mean()),
                "mean_diff": float(region_diff.mean()),
                "timing_accuracy": (
                    "PAPI more precise" if region_diff.mean() > 0 else "Wall clock more precise"
                ),
            }

        return analysis

    def print_timing_summary(self) -> None:
        """Print a summary of timing analysis."""
        analysis = self.get_timing_analysis()

        if "error" in analysis:
            # print(f"Timing Analysis Error: {analysis['error']}")
            return

        # print("\n=== Timing Analysis Summary ===")
        # print(f"Total measurements: {analysis['total_measurements']}")

        # print(f"\nWall Clock Timing:")
        ws = analysis["wall_clock_stats"]
        # Fix dictionary key access errors by ensuring proper type handling
        if isinstance(ws, dict):
            # print(f"  Mean: {ws.get('mean', 0):.6f}s ± {ws.get('std', 0):.6f}s")
            # print(f"  Range: {ws.get('min', 0):.6f}s - {ws.get('max', 0):.6f}s")
            pass

        # print(f"\nPAPI High-Resolution Timing (from microseconds):")
        ps = analysis["papi_stats"]
        if isinstance(ps, dict):
            # print(f"  Mean: {ps.get('mean', 0):.6f}s ± {ps.get('std', 0):.6f}s")
            # print(f"  Range: {ps.get('min', 0):.6f}s - {ps.get('max', 0):.6f}s")
            pass

        # print(f"\nTiming Comparison:")
        td = analysis["timing_difference_stats"]
        if isinstance(td, dict):
            # print(f"  Mean difference: {td.get('mean_diff', 0):.6f}s")
            # print(f"  Max absolute difference: {td.get('max_abs_diff', 0):.6f}s")
            # print(f"  Correlation: {td.get('correlation', 0):.4f}")
            if abs(td.get("mean_diff", 0)) < 0.001:
                # print("  ✓ Good agreement between timing methods")
                pass
            else:
                # print("  ⚠ Significant difference between timing methods")
                pass

        # print(f"\nBy Region:")
        # Ensure `analysis['by_region']` is treated as a dictionary before
        # accessing `.items()`
        by_region = analysis.get("by_region", {})
        if isinstance(by_region, dict):
            for _region, _data in by_region.items():
                # print(f"  {region}:")
                # print(f"    Wall clock: {data.get('wall_clock_mean', 0):.6f}s")
                # print(f"    PAPI: {data.get('papi_mean', 0):.6f}s")
                # print(f"    Difference: {data.get('mean_diff', 0):.6f}s ({data.get('timing_accuracy', 'Unknown')})")
                pass
        else:
            # print("Error: 'by_region' is not a dictionary")
            pass

    def start_epoch(self, epoch_num: int) -> None:
        """Mark the start of a new epoch."""
        self.current_epoch = epoch_num
        self.current_batch = 0
        logger.info("Starting epoch %s/%s", epoch_num + 1, self.max_epochs)

    def end_epoch(self, epoch_num: int) -> None:
        """Mark the end of an epoch."""
        logger.info(
            "Completed epoch %s/%s with %s batches",
            epoch_num + 1,
            self.max_epochs,
            self.current_batch,
        )

    def next_epoch(self) -> None:
        """Advance to the next epoch and reset batch counter."""
        self.current_epoch += 1
        self.current_batch = 0
        if self.current_epoch % 5 == 0:  # Log progress every 5 epochs
            logger.info(
                "Profiling progress: Epoch %s/%s",
                self.current_epoch,
                self.max_epochs,
            )

    def next_batch(self) -> None:
        """Advance to the next batch."""
        self.current_batch += 1
        if self.batch_based_cycling:
            # Cycle through event sets every batch
            self.current_event_set_idx = (self.current_event_set_idx + 1) % len(
                self.event_sets,
            )

    def should_continue_profiling(self) -> bool:
        """Check if profiling should continue based on epoch count."""
        return self.current_epoch < self.max_epochs

    def get_profiling_status(self) -> dict[str, Any]:
        """Get comprehensive profiling status including epoch and batch info."""
        return {
            "current_epoch": self.current_epoch,
            "current_batch": self.current_batch,
            "max_epochs": self.max_epochs,
            "epoch_progress": f"{self.current_epoch + 1}/{self.max_epochs}",
            "batch_based_cycling": self.batch_based_cycling,
            "current_event_set": self.current_event_set_idx,
            "total_event_sets": len(self.event_sets),
            "event_set_progress": (
                f"{self.current_event_set_idx + 1}/{len(self.event_sets)}"
                if self.event_sets
                else "0/0"
            ),
            "should_continue": self.should_continue_profiling(),
            "epochs_remaining": max(0, self.max_epochs - self.current_epoch),
        }
