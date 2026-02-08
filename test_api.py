"""
Quick API tester - tests model with random samples from each class
"""
import requests
import os
import random
from pathlib import Path

API_URL = "http://localhost:8000/analyze"
DATASET_PATH = "dataset/PRoject"

def test_class(class_name, num_samples=5):
    """Test API with random samples from a class"""
    class_dir = Path(DATASET_PATH) / class_name
    images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        print(f"[X] No images found in {class_name}")
        return None

    # Random sample
    samples = random.sample(images, min(num_samples, len(images)))

    correct = 0
    results = []

    print(f"\n{'='*70}")
    print(f"Testing: {class_name} ({num_samples} samples)")
    print(f"{'='*70}")

    for img_name in samples:
        img_path = class_dir / img_name

        with open(img_path, 'rb') as f:
            files = {'file': (img_name, f, 'image/jpeg')}
            response = requests.post(API_URL, files=files, timeout=30)

        if response.status_code == 200:
            result = response.json()
            predicted = result['prediction']['class_name']
            confidence = result['prediction']['confidence']

            is_correct = predicted == class_name
            correct += is_correct

            status = "[OK]" if is_correct else "[X]"
            print(f"  {status} {img_name[:30]:30s} -> {predicted:30s} ({confidence:.1f}%)")

            results.append({
                'image': img_name,
                'expected': class_name,
                'predicted': predicted,
                'confidence': confidence,
                'correct': is_correct
            })
        else:
            print(f"  [X] API Error: {response.status_code}")

    accuracy = (correct / num_samples) * 100 if num_samples > 0 else 0
    print(f"\nAccuracy: {correct}/{num_samples} = {accuracy:.1f}%")

    return {
        'class': class_name,
        'accuracy': accuracy,
        'correct': correct,
        'total': num_samples,
        'results': results
    }

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" SOLAR PANEL FAULT DETECTION - API TESTING")
    print("="*70)

    # Test each class
    classes = [
        "Bird_drop_generateds",
        "Clean",
        "Dusty",
        "Electrical_damage_generated",
        "Physcial_damage_generated",
        "Snow_covered_generated"
    ]

    all_results = []

    for class_name in classes:
        result = test_class(class_name, num_samples=10)
        if result:
            all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(" SUMMARY")
    print(f"{'='*70}")

    total_correct = sum(r['correct'] for r in all_results)
    total_samples = sum(r['total'] for r in all_results)
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0

    for r in all_results:
        print(f"  {r['class']:35s} -> {r['accuracy']:5.1f}% ({r['correct']}/{r['total']})")

    print(f"\n  {'OVERALL ACCURACY':35s} -> {overall_accuracy:5.1f}% ({total_correct}/{total_samples})")
    print(f"{'='*70}\n")

    # Decision
    if overall_accuracy >= 90:
        print("[OK] MODEL PERFORMS WELL - READY FOR DELIVERY!")
    elif overall_accuracy >= 85:
        print("[!] MODEL IS ACCEPTABLE - Consider minor improvements")
    else:
        print("[X] MODEL NEEDS IMPROVEMENT - Try ResNet34 or more training")
