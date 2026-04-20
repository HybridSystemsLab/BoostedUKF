# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Main pipeline:
1. LRW
2. WFM
3. EKF
4. UKF
5. EnKF
6. Boosted UKF
"""
import time
import subprocess
import sys
from glob import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE_STEPS = [
    ("lrw.py",        "LRW"),
    ("wfm.py",        "WFM"),
    ("ekf.py",        "EKF"),
    ("ukf.py",        "UKF"),
    ("enkf.py",       "EnKF"),
    ("boosted_ukf.py","Boosted UKF"),
]


def print_progress(step_idx, total_steps, step_name):
    percent_start = (step_idx / total_steps) * 100
    print("\n" + "=" * 70)
    print(f"Step {step_idx + 1}/{total_steps}: {step_name}")
    print(f"Pipeline progress: {percent_start:.1f}%")
    print("=" * 70)


def run_script(script_name, step_name, step_idx, total_steps):
    print_progress(step_idx, total_steps, step_name)
    script_path = os.path.join(BASE_DIR, script_name)
    start_time = time.time()

    # Run each script in its own subprocess to avoid OpenMP/DLL conflicts
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
        check=True
    )

    elapsed = time.time() - start_time
    percent_end = ((step_idx + 1) / total_steps) * 100
    print(f"\nFinished {step_name} in {elapsed:.1f} seconds.")
    print(f"Pipeline progress: {percent_end:.1f}%")


def check_files_exist(file_list, step_name):
    missing = [f for f in file_list if not os.path.exists(os.path.join(BASE_DIR, f))]
    if missing:
        print(f"\nMissing files after {step_name}:")
        for f in missing:
            print(f"  - {f}")
        raise FileNotFoundError(f"{step_name} did not generate all required files.")
    print(f"All required output files from {step_name} exist.")


def list_saved_figures():
    print("\n" + "=" * 70)
    print("Saved figures")
    print("=" * 70)
    folders = [
        "lrw_results",
        "wfm_results",
        "ekf_results",
        "ukf_results",
        "enkf_results",
        "boosted_ukf_results",
    ]
    exts = ("*.png", "*.jpg", "*.jpeg", "*.pdf", "*.svg")
    found = False
    for folder in folders:
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        files = []
        for ext in exts:
            files.extend(glob(os.path.join(folder_path, ext)))
        if files:
            found = True
            print(f"\n{folder}:")
            for f in sorted(files):
                print("  -", os.path.basename(f))
    if not found:
        print("No figure files found.")


def main():
    total_steps = len(PIPELINE_STEPS)

    # Step 1: LRW
    run_script("lrw.py", "LRW", 0, total_steps)
    check_files_exist([
        "lrw_results/training_data_weights_noise_0.01.xlsx",
        "lrw_results/training_data_weights_noise_0.001.xlsx",
        "lrw_results/training_data_weights_noise_0.0001.xlsx",
    ], "LRW")

    # Step 2: WFM
    run_script("wfm.py", "WFM", 1, total_steps)
    check_files_exist([
        "lrw_results/gen10k_noise_0.01_gen.xlsx",
        "lrw_results/gen10k_noise_0.001_gen.xlsx",
        "lrw_results/gen10k_noise_0.0001_gen.xlsx",
    ], "WFM")

    # Step 3: EKF
    run_script("ekf.py", "EKF", 2, total_steps)

    # Step 4: UKF
    run_script("ukf.py", "UKF", 3, total_steps)

    # Step 5: EnKF
    run_script("enkf.py", "EnKF", 4, total_steps)

    # Step 6: Boosted UKF
    run_script("boosted_ukf.py", "Boosted UKF", 5, total_steps)

    print("\n" + "=" * 70)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("Pipeline progress: 100.0%")
    print("=" * 70)

    list_saved_figures()


if __name__ == "__main__":
    main()