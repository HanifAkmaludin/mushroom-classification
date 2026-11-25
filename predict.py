"""
Inference script for predicting mushroom edibility given physical characteristics.
Includes automated botanical food safety checks and warnings.
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from src.config import MODELS_DIR, VALUE_MAPPINGS
from src.models import load_model


def get_sample_mushrooms():
    """
    Returns realistic sample mushroom profiles for testing inference.
    """
    samples = {
        "Sample 1 (Almond Odor - Expected Edible)": {
            "cap_shape": "convex",
            "cap_surface": "smooth",
            "cap_color": "yellow",
            "bruises": "bruises",
            "odor": "almond",
            "gill_attachment": "free",
            "gill_spacing": "close",
            "gill_size": "broad",
            "gill_color": "black",
            "stalk_shape": "enlarging",
            "stalk_root": "club",
            "stalk_surface_above_ring": "smooth",
            "stalk_surface_below_ring": "smooth",
            "stalk_color_above_ring": "white",
            "stalk_color_below_ring": "white",
            "veil_color": "white",
            "ring_number": "one",
            "ring_type": "pendant",
            "spore_print_color": "brown",
            "population": "numerous",
            "habitat": "woods",
        },
        "Sample 2 (Foul Odor - Expected Poisonous)": {
            "cap_shape": "convex",
            "cap_surface": "scaly",
            "cap_color": "white",
            "bruises": "no",
            "odor": "foul",
            "gill_attachment": "free",
            "gill_spacing": "close",
            "gill_size": "narrow",
            "gill_color": "black",
            "stalk_shape": "enlarging",
            "stalk_root": "bulbous",
            "stalk_surface_above_ring": "silky",
            "stalk_surface_below_ring": "silky",
            "stalk_color_above_ring": "white",
            "stalk_color_below_ring": "white",
            "veil_color": "white",
            "ring_number": "one",
            "ring_type": "pendant",
            "spore_print_color": "chocolate",
            "population": "several",
            "habitat": "paths",
        },
        "Sample 3 (Green Spores - Toxic Amanita/Lepiota)": {
            "cap_shape": "flat",
            "cap_surface": "smooth",
            "cap_color": "brown",
            "bruises": "no",
            "odor": "pungent",
            "gill_attachment": "free",
            "gill_spacing": "close",
            "gill_size": "narrow",
            "gill_color": "black",
            "stalk_shape": "enlarging",
            "stalk_root": "bulbous",
            "stalk_surface_above_ring": "silky",
            "stalk_surface_below_ring": "silky",
            "stalk_color_above_ring": "white",
            "stalk_color_below_ring": "white",
            "veil_color": "white",
            "ring_number": "one",
            "ring_type": "large",
            "spore_print_color": "green",
            "population": "several",
            "habitat": "urban",
        },
    }
    return samples


def predict_sample(sample_dict: dict, model_type: str = "rf", threshold: float = 0.5):
    """
    Predicts mushroom edibility from raw dictionary of botanical attributes.
    """
    model_file = "random_forest.joblib" if model_type == "rf" else "decision_tree.joblib"
    metadata_file = MODELS_DIR / "model_metadata.joblib"

    if not metadata_file.exists():
        print(f"Error: Model metadata not found at {metadata_file}. Please run train.py first.")
        sys.exit(1)

    metadata = joblib.load(metadata_file)
    model = load_model(model_file)
    expected_features = metadata["feature_names"]

    # Convert sample dict to DataFrame
    df_sample = pd.DataFrame([sample_dict])
    
    # One-hot encode with pandas get_dummies
    df_encoded = pd.get_dummies(df_sample)
    
    # Reindex to match trained features
    df_aligned = df_encoded.reindex(columns=expected_features, fill_value=0)

    # Predict probability
    probs = model.predict_proba(df_aligned)[0]
    prob_poisonous = probs[1]
    
    # Decision with safety threshold
    is_poisonous = prob_poisonous >= threshold
    
    return {
        "model": "Random Forest" if model_type == "rf" else "Decision Tree",
        "prob_poisonous": prob_poisonous,
        "prob_edible": probs[0],
        "is_poisonous": bool(is_poisonous),
        "safety_threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Mushroom Edibility & Botanical Toxicity Predictor")
    parser.add_argument("--model", choices=["rf", "dt"], default="rf", help="Model to use (rf or dt)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for poisonous class")
    args = parser.parse_args()

    print("=" * 65)
    print(" BOTANICAL MUSHROOM SAFETY INFERENCE")
    print(f" Selected Model: {'Random Forest' if args.model == 'rf' else 'Decision Tree'}")
    print(f" Safety Threshold: {args.threshold} (Scores >= {args.threshold} flagged as POISONOUS)")
    print("=" * 65)

    samples = get_sample_mushrooms()
    for name, sample in samples.items():
        print(f"\nEvaluating: {name}")
        print("Key Characteristics:")
        print(f"  - Odor: {sample['odor']} | Gill Spacing: {sample['gill_spacing']} | Spore Color: {sample['spore_print_color']}")
        
        result = predict_sample(sample, model_type=args.model, threshold=args.threshold)
        p_poison = result["prob_poisonous"] * 100
        
        if result["is_poisonous"]:
            print(f"  [!] PREDICTION: POISONOUS / TOXIC (P(Poisonous) = {p_poison:.1f}%)")
            print("      CAUTION: DO NOT CONSUME! High botanical toxicity risk.")
        else:
            print(f"  [v] PREDICTION: EDIBLE (P(Poisonous) = {p_poison:.1f}%)")
            print("      Safe to consume according to physical characteristics.")


if __name__ == "__main__":
    main()
