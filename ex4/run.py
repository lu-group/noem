from pathlib import Path
from ex4_1 import main as ex4_1_main
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
model_name = "ex4.1"
ex4_1_model_path = PROJECT_ROOT / "data_driven_training" / "ex4_darcy" / f"{model_name}.pth"
ex4_1_main.run(net_name=str(ex4_1_model_path))

from ex4_4 import main as ex4_4_main
data_path = "ex4_4//training_data//"
data_set_name = "ex4.4"
# ex4_4_gendata.ex4_4_data_gen()
model_name = "ex4_4"
from data_driven_training.nonlinear_darcy_training import train_model as nonlinear_darcy_model_training
# nonlinear_darcy_model_training(data_path=data_path, sample_name=data_set_name, model_path=model_path, model_name=model_name)
ex4_4_model_path = PROJECT_ROOT / "data_driven_training" / "ex4_darcy" / f"{model_name}.pth"
ex4_4_main.run(str(ex4_4_model_path))