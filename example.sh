# treemodel
python run_experiment.py --dataset electricity --model CatBoost --task single
python run_experiment.py --dataset iris --model XGBoost --task least
python run_experiment.py --dataset iris --model LightGBM --task random

# dlmodel
CUDA_VISIBLE_DEVICES=0 python run_experiment.py --dataset iris --model TabPFN --task random
CUDA_VISIBLE_DEVICES=0 python run_experiment.py --dataset iris --model TabPFN --task single
CUDA_VISIBLE_DEVICES=0 python run_experiment.py --dataset electricity --model mlp --task most

# LLMs
python run_experiment.py --dataset credit --model Llama3-8B --task single
python run_experiment.py --dataset credit --model Llama3-8B --task least
python run_experiment.py --dataset credit --model Llama3-8B --task most
python run_experiment.py --dataset credit --model Llama3-8B --task random
python run_experiment.py --dataset credit --model Llama3-8B --task random --degree 0.1

# tabular LLMs
CUDA_VISIBLE_DEVICES=0 python run_experiment.py --dataset electricity --model TabLLM --task single
CUDA_VISIBLE_DEVICES=0 python run_experiment.py --dataset iris --model UniPredict --task random