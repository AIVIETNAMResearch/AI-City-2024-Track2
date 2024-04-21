cd src/train/prepare_train_data
python prepare_train_data_external.py --choice segment
python prepare_train_data_external.py --choice rewrite

python prepare_train_data_internal_vehicle_view.py --choice segment
python prepare_train_data_internal_vehicle_view.py --choice rewrite

python prepare_train_data_internal_overhead_view.py --choice segment
python prepare_train_data_internal_overhead_view.py --choice rewrite