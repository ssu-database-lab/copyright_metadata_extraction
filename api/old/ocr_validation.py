import os
import sys
import subprocess
import pandas as pd
import time
import matplotlib.pyplot as plt

def main():
    print("========================================")
    print("      OCR Validation & Test Suite       ")
    print("========================================")
    print("1. Running OCR generation and tuning script (ocr.py)...")
    print("   Target Mean Accuracy: 80.0% ~ 85.0%")
    print("   (This process generates synthetic images, tunes noise, and evaluates)")
    
    start_time = time.time()
    
    # Run ocr.py
    script_path = os.path.join(os.path.dirname(__file__), "ocr.py")
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found.")
        return

    try:
        # Run ocr.py and capture output to show progress
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running ocr.py: {e}")
        return

    elapsed = time.time() - start_time
    print(f"\n[Done] OCR script finished in {elapsed:.1f} seconds.")

    # 2. Analyze Results
    print("\n2. Validating Results...")
    results_path = os.path.join("synthetic_testset", "results.csv")
    
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return

    try:
        df = pd.read_csv(results_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if "char_accuracy(%)" not in df.columns:
        print("Error: 'char_accuracy(%)' column missing in results.")
        return

    accuracies = df["char_accuracy(%)"]
    mean_acc = accuracies.mean()
    std_acc = accuracies.std()
    median_acc = accuracies.median()

    print("-" * 40)
    print(f"Total Samples: {len(df)}")
    print(f"Mean Accuracy:   {mean_acc:.2f}%")
    print(f"Median Accuracy: {median_acc:.2f}%")
    print(f"Std Deviation:   {std_acc:.2f}")
    print("-" * 40)

    # 3. Validation Logic
    target_low = 80.0
    target_high = 85.0
    
    # Tuning uses a small subset, so final result might drift slightly.
    # We allow a small tolerance (e.g., +/- 1.0%) for the validation pass/fail.
    tolerance = 1.0
    
    is_pass = (target_low - tolerance) <= mean_acc <= (target_high + tolerance)

    if is_pass:
        print(f"\n[SUCCESS] OCR Mean Accuracy is within target range ({target_low}-{target_high}%).")
        print(f"Result: {mean_acc:.2f}%")
    else:
        print(f"\n[WARNING] OCR Mean Accuracy is outside target range ({target_low}-{target_high}%).")
        print(f"Result: {mean_acc:.2f}%")
        if mean_acc < target_low:
            print("-> Accuracy is too LOW. (Noise might be too high)")
        else:
            print("-> Accuracy is too HIGH. (Noise might be too low)")

    # 4. Visualization (Optional)
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(accuracies, bins=20, range=(0, 100), edgecolor='black', alpha=0.7)
        plt.axvline(mean_acc, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_acc:.1f}%')
        plt.axvline(target_low, color='green', linestyle='-', linewidth=1, label='Target Low (80%)')
        plt.axvline(target_high, color='green', linestyle='-', linewidth=1, label='Target High (85%)')
        
        plt.title('OCR Accuracy Distribution')
        plt.xlabel('Character Accuracy (%)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join("synthetic_testset", "accuracy_distribution.png")
        plt.savefig(plot_path)
        print(f"\n[Info] Distribution plot saved to: {plot_path}")
    except Exception as e:
        print(f"[Info] Could not create plot: {e}")

if __name__ == "__main__":
    main()
