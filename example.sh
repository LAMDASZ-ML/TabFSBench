# LLM
python run_experiment.py --dataset credit --model Llama3-8B --task single
python run_experiment.py --dataset credit --model Llama3-8B --task multi-removeleast
python run_experiment.py --dataset credit --model Llama3-8B --task multi-removemost
python run_experiment.py --dataset credit --model Llama3-8B --task random
python run_experiment.py --dataset credit --model Llama3-8B --task random --degree 0.1

# treemodel
python run_experiment.py --dataset electricity --model CatBoost --task single
python run_experiment.py --dataset iris --model XGBoost --task multi-removeleast
python run_experiment.py --dataset iris --model XGBoost --task multi-removemost
python run_experiment.py --dataset concrete --model LightGBM --task random
python run_experiment.py --dataset concrete --model LightGBM --task random --degree 0.1

#dlmodel
CUDA_VISIBLE_DEVICES=0 python run_experiment.py --dataset iris --model TabPFN --task random
CUDA_VISIBLE_DEVICES=0 python run_experiment.py --dataset iris --model TabPFN --task single