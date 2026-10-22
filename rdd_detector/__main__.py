import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Experimental RDD image scoring (no calibrated probability)")
    parser.add_argument("image")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional experimental cutoff; source notebook used 0.0001")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-samples", type=int, default=16)
    parser.add_argument("--noise-batch-size", type=int, default=1)
    args = parser.parse_args()
    from .detector import RDDDetector
    detector = RDDDetector(device=args.device, seed=args.seed,
                           noise_samples=args.noise_samples, noise_batch_size=args.noise_batch_size)
    print(json.dumps(detector.score(args.image, args.threshold).to_dict(), indent=2))


if __name__ == "__main__":
    main()

