import sys
import threading
import time
from pathlib import Path

import tqdm
import yaml

# NARX-MPC/COBR/ contains data_sampling/ and models/ — add it to the path
# so imports work regardless of which subdirectory this script lives in.
COBR_ROOT = Path(__file__).resolve().parent.parent / "COBR"
sys.path.insert(0, str(COBR_ROOT))

from data_sampling.helpers.simulation_sampler import SimulationSampler
from models.base_cobr_model import get_base_COBR_model  # type: ignore[import]


def load_configs(
    sampling_config_path=None,
    model_config_path=None,
):
    """Load sampling and model configs from YAML files.

    Defaults:
        sampling_config_path — <this file's dir>/configs/thermal_cobr_sampling_config.yaml
        model_config_path    — COBR/models/configs/thermal_cobr_config.yaml

    Pass explicit Path or str arguments to load configs from any directory
    under NARX-MPC (e.g. a different Run_* folder or the COBR tree).
    """
    if sampling_config_path is None:
        sampling_config_path = Path(__file__).parent / "configs" / "thermal_cobr_sampling_config.yaml"
    if model_config_path is None:
        model_config_path = Path(__file__).parent / "configs" / "thermal_cobr_config.yaml"

    sampling_config_path = Path(sampling_config_path)
    model_config_path = Path(model_config_path)

    with sampling_config_path.open("r", encoding="utf-8") as f:
        sampling_config = yaml.safe_load(f)
    with model_config_path.open("r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    return sampling_config, model_config

#Load the COBR Model
def create_reactor_model(model_config):
    model = get_base_COBR_model(model_config)
    model.setup()
    return model


def main():
    
    # File sampling parameters and set up the output directory
    # 1 sample - One complete simulation run with a unique x_0
    n_samples = 500
    n_workers = 17
    from datetime import date
    date_str = date.today().strftime("%d_%m_%Y")
    data_set_path = Path(__file__).parent / "data_sets" / date_str
    data_set_path.mkdir(parents=True, exist_ok=True)
    h5_file = data_set_path / "thermal_cobr_raw_data.h5"

    print(f"Running {n_samples} simulations with {n_workers} workers...")

    sampling_config, model_config = load_configs()

    sampler = SimulationSampler(
        model_creation_func=create_reactor_model,
        sampling_config=sampling_config,
        h5_file_path=str(h5_file),
        n_samples=n_samples,
        n_workers=n_workers,
        model_config=model_config,
    )

    pbar = tqdm.tqdm(total=n_samples, unit='sim', desc='Simulations', dynamic_ncols=True)
    _last = [0]
    _stop = threading.Event()

    def _monitor():
        while not _stop.is_set():
            try:
                done = sampler._get_next_simulation_id()
                if done > _last[0]:
                    pbar.update(done - _last[0])
                    _last[0] = done
            except Exception:
                pass
            _stop.wait(1.0)

    _thread = threading.Thread(target=_monitor, daemon=True)
    _thread.start()

    start_time = time.time()
    summary = sampler.run()
    duration = time.time() - start_time

    _stop.set()
    _thread.join()
    done = summary['completed']
    if done > _last[0]:
        pbar.update(done - _last[0])
    pbar.close()

    print(f"Completed {summary['completed']} simulations in {duration:.1f}s")
    if summary['completed'] > 0:
        print(f"Time/sim: {duration / summary['completed']:.2f}s  |  Rate: {summary['completed'] / duration:.2f} sim/s")
    print(f"Saved dataset to {h5_file}")


if __name__ == "__main__":
    main()
