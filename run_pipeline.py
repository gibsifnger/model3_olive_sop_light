from src.step01_generate_synthetic_data import generate_all_input_data
from src.step02_build_features import build_feature_table
from src.step03_train_forecast_model import train_and_select_forecast_model
from src.step04_build_allocation_plan import build_allocation_plan
from src.step05_decide_replenishment_action import decide_replenishment_action
from src.utils import build_summary_table


def main() -> None:
    generate_all_input_data()
    build_feature_table()
    train_and_select_forecast_model()
    build_allocation_plan()
    decide_replenishment_action()
    build_summary_table()


if __name__ == "__main__":
    main()
