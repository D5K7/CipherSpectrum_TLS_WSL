from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cipherspectrum_tls13.settings import load_config
from cipherspectrum_tls13.train_eval import cross_cipher_eval, train_one_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CipherSpectrum TLS1.3 classifier")
    parser.add_argument("--config", type=str, default="configs/default_experiment.yaml")
    parser.add_argument("--cross-cipher", action="store_true", help="Run cross-cipher evaluation after training")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuilding dataset index")
    args = parser.parse_args()

    config = load_config(args.config)
    result = train_one_run(config, force_index_rebuild=args.rebuild_index)
    print(json.dumps(result, indent=2))

    if args.cross_cipher:
        ckpt = Path(config.output.output_dir) / config.output.run_name / "best_model.pt"
        cc_result = cross_cipher_eval(config, checkpoint_path=ckpt)
        print(json.dumps(cc_result, indent=2))


if __name__ == "__main__":
    main()
