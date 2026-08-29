import os
import shutil
import sys
import tempfile
import time

import nibabel as nib
import numpy as np
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from mriforge.data.io_strategies import DicomStrategy, NiftiStrategy


def create_dummy_dicom_series(dir_path, num_slices=30):
    print(f"Creating {num_slices} dummy DICOM files in {dir_path}...")
    for i in range(num_slices):
        filename = os.path.join(dir_path, f"slice_{i:03d}.dcm")
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
        file_meta.MediaStorageSOPInstanceUID = f"1.2.3.{i}"
        file_meta.ImplementationClassUID = "1.2.3.4"
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.PatientName = "Test^Patient"
        ds.PatientID = "123456"
        ds.InstanceNumber = i + 1
        ds.SliceLocation = float(i)
        ds.Rows = 320
        ds.Columns = 320
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        # Random data
        data = np.random.randint(0, 1000, (320, 320), dtype=np.uint16)
        ds.PixelData = data.tobytes()
        ds.save_as(filename)


def create_dummy_nifti(file_path, shape=(320, 320, 30)):
    print(f"Creating dummy NIfTI file {file_path} with shape {shape}...")
    data = np.random.randn(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, file_path)


def benchmark_dicom_strategy():
    print("\nBenchmarking DicomStrategy...")
    temp_dir = tempfile.mkdtemp()
    try:
        create_dummy_dicom_series(temp_dir, num_slices=50)

        strategy = DicomStrategy()

        # Benchmark Full Series Load (Current Behavior)
        start_time = time.time()
        for _ in range(5):
            _ = strategy.load(temp_dir)
        end_time = time.time()
        print(f"  Load Full Series (avg of 5): {(end_time - start_time) / 5:.4f} s")

        # Benchmark Slice Load (Proposed Optimization)
        # Assuming we implement slice loading via metadata={'slice_index': i}
        # For now, this will fallback to full load until optimized, so it serves as baseline
        start_time = time.time()
        for _ in range(5):
            _ = strategy.load(temp_dir, metadata={"slice_index": 25})
        end_time = time.time()
        print(
            f"  Load Single Slice [Baseline] (avg of 5): {(end_time - start_time) / 5:.4f} s"
        )

    finally:
        shutil.rmtree(temp_dir)


def benchmark_nifti_strategy():
    print("\nBenchmarking NiftiStrategy...")
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        tmp_path = f.name

    try:
        create_dummy_nifti(tmp_path, shape=(320, 320, 100))  # 100 slices

        strategy = NiftiStrategy()

        # Benchmark Full Volume Load
        start_time = time.time()
        for _ in range(10):
            _ = strategy.load(tmp_path)
        end_time = time.time()
        print(f"  Load Full Volume (avg of 10): {(end_time - start_time) / 10:.4f} s")

        # Benchmark Slice Load
        start_time = time.time()
        for _ in range(100):
            _ = strategy.load(tmp_path, metadata={"slice_index": 50})
        end_time = time.time()
        print(
            f"  Load Single Slice (avg of 100): {(end_time - start_time) / 100:.4f} s"
        )

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    benchmark_dicom_strategy()
    benchmark_nifti_strategy()
