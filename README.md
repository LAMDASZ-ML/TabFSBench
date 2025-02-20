<div align="center">
    <h1>
        TabFSBench: Tabular Benchmark for Feature Shifts in Open Environment
    </h1>
</div>
<div align="center">
<p>
        <a href="https://arxiv.org/abs/2501.18935">[Paper]</a> &nbsp;&nbsp;&nbsp; <a href="https://github.com/LAMDASZ-ML/TabFSBench">[Code]</a> &nbsp;&nbsp;&nbsp; <a href="https://clementcheng0217.github.io/Tab-index/">[Project page]</a>
    <p>
</div>




---

## Introduction

**TabFSBench** is a benchmarking tool for feature shifts in tabular data in open-environment scenarios. It aims to analyse the performance and robustness of a model in feature shifts.

**TabFSBench** offers the following advantages:

- **Various Models**: Tree-based models, deep-learning models, LLMs and tabular LLMs.
- **Diverse Experiments**: Single shift, most/least-revelant shift and random shift.
- **Exportable Datasets**: Be able to export the feature-shift version of the given dataset.
- **Addable Components**: Supports to add new datasets and models, and export the given dataset under the specific experiment.

**If you use the benchmark in your research, please cite the paper:**

```bibtex

@article{cheng2025tabfsbenchtabularbenchmarkfeature,

      title={TabFSBench: Tabular Benchmark for Feature Shifts in Open Environment},

      author={Zi-Jian Cheng and Zi-Yi Jia and Zhi Zhou and Lan-Zhe Guo and Yu-Feng Li},

      journal={arXiv preprint arXiv:2501.18935},

      year={2025}
}

```

<section class="section" id="News">
    <div class="container is-max-desktop content">
      <h2 class="title">News</h2>
      <div style="border:1px solid #CCC"></div>  
      <ul>
      <li>[2025-02] Our <a href="https://clementcheng0217.github.io/Tab-index/" target="_blank">project page</a> is released. </li>
      <li>[2025-01] Our <a href="https://github.com/LAMDASZ-ML/TabFSBench" target="_blank">code</a> is available now. </li>
      <li>[2025-01] Our <a href="https://arxiv.org/abs/2501.18935" target="_blank">paper</a> is accessible now. </li>
      </ul>
      <p>If you have any questions, please contact us at chengzj@lamda.nju.edu.cn or submit an issue in the project <a href="https://github.com/LAMDASZ-ML/TabFSBench">issue</a>.</p>
    </div>
  </section>

## Quickstart

### 1. Download

Download this GitHub repository.

```bash
git clone https://github.com/LAMDASZ-ML/TabFSBench.git
cd TabFSBench
```

### 2. Environment setup

Create a new Python 3.10 environment and install 'requirements.txt'.

```bash
conda create --name tabfsbench python=3.10
pip install -r requirements.txt
```
### 3. Run
You need to input four parameters to use TabFSBench. There are dataset, model, task and degree.

**dataset** and **model**: input the full name. 

**task**: You can choose 'single', 'least', 'most' or 'random' as TaskName.

**degree**: Degree refers to the number of missing columns as a percentage of the total number of columns in the dataset, in the range 0-1. If you want to see the performance of the model at all missing degrees, set Degree to 'all'.

**export_dataset**: Whether to export the dataset or not. Default is 'False'.
```bash
python run_experiment.py --dataset DatasetName --model ModelName --task TaskName --degree Degree --export_dataset True/False
```

In **example.sh** you can get different kinds of instruction samples.

## Benchmark Datasets

All the datasets used in TabFSBench are publicly available. You can get them from [OpenML](https://www.openml.org/) or [Kaggle](https://www.kaggle.com/). Also you can directly use them from `./datasets`.

### How to Add New Datasets

Datasets used in TabFSBench are placed in the project's current directory, corresponding to the file name.

Each dataset folder consists of:

- `dataset.csv`, which must be included.

- `info.json`, which must include the following two contents (task can be "regression", "multiclass" or "binary", link can be from Kaggle or OpenML, num_classes is optional):
  

  ```json
  {
    "task": "binary", 
    "link": "www.kaggle.com",
    "num_classes":
  }
  ```


## Models

TabFSBench is possible to test three kinds of models' performance directly, including tree-based models, deep learning models and tabular LLMs. For LLMs, TabFSBnech provides text files(.json) about the given dataset that can be used directly for LLM to finetune.

#### Tree-based models
1. **[CatBoost](https://catboost.ai/)**: A powerful boosting-based model designed for efficient handling of categorical features.
2. **[LightGBM](https://lightgbm.readthedocs.io/en/latest/index.html)**: A machine-learning model based on the Boosting algorithm.
3. **[XGBoost](https://xgboost.readthedocs.io/en/latest/index.html)**: A machine-learning model incrementally building multiple decision trees by optimizing the loss function.

#### Deep learning models
We use LAMDA-TALENT to evaluate deep-learning models. You can get details from **[LAMDA-TALENT](https://github.com/qile2000/LAMDA-TALENT)**.
1. **MLP**: A multi-layer neural network, which is implemented according to [RTDL](https://arxiv.org/abs/2106.11959).
2. **ResNet**: A DNN that uses skip connections across many layers, which is implemented according to [RTDL](https://arxiv.org/abs/2106.11959).
3. **[SNN](https://arxiv.org/abs/1706.02515)**: An MLP-like architecture utilizing the SELU activation, which facilitates the training of deeper neural networks.
4. **[DANets](https://arxiv.org/abs/2112.02962)**: A neural network designed to enhance tabular data processing by grouping correlated features and reducing computational complexity.
5. **[TabCaps](https://openreview.net/pdf?id=OgbtSLESnI)**: A capsule network that encapsulates all feature values of a record into vectorial features.
6. **[DCNv2](https://arxiv.org/abs/2008.13535)**: Consists of an MLP-like module combined with a feature crossing module, which includes both linear layers and multiplications.
7. **[NODE](https://arxiv.org/abs/1909.06312)**: A tree-mimic method that generalizes oblivious decision trees, combining gradient-based optimization with hierarchical representation learning.
8. **[GrowNet](https://arxiv.org/abs/2002.07971)**: A gradient boosting framework that uses shallow neural networks as weak learners.
9. **[TabNet](https://arxiv.org/abs/1908.07442)**: A tree-mimic method using sequential attention for feature selection, offering interpretability and self-supervised learning capabilities.
10. **[TabR](https://arxiv.org/abs/2307.14338)**: A deep learning model that integrates a KNN component to enhance tabular data predictions through an efficient attention-like mechanism.
11. **[ModernNCA](https://arxiv.org/abs/2407.03257)**: A deep tabular model inspired by traditional Neighbor Component Analysis, which makes predictions based on the relationships with neighbors in a learned embedding space.
12. **[AutoInt](https://arxiv.org/abs/1810.11921)**: A token-based method that uses a multi-head self-attentive neural network to automatically learn high-order feature interactions.
13. **[Saint](https://arxiv.org/abs/2106.01342)**: A token-based method that leverages row and column attention mechanisms for tabular data.
14. **[TabTransformer](https://arxiv.org/abs/2012.06678)**: A token-based method that enhances tabular data modeling by transforming categorical features into contextual embeddings.
15. **[FT-Transformer](https://arxiv.org/abs/2106.11959)**: A token-based method which transforms features to embeddings and applies a series of attention-based transformations to the embeddings.
16. **[TANGOS](https://openreview.net/pdf?id=n6H86gW8u0d)**: A regularization-based method for tabular data that uses gradient attributions to encourage neuron specialization and orthogonalization.
17. **[SwitchTab](https://arxiv.org/abs/2401.02013)**: A self-supervised method tailored for tabular data that improves representation learning through an asymmetric encoder-decoder framework. Following the original paper, our toolkit uses a supervised learning form, optimizing both reconstruction and supervised loss in each epoch.
18. **[TabPFN](https://arxiv.org/abs/2207.01848)**: A general model which involves the use of pre-trained deep neural networks that can be directly applied to any tabular task. TabFSBench uses the first version of TabPFN and supports to evaluate [TabPFNv2](https://www.nature.com/articles/s41586-024-08328-6) by updating the version. 

#### LLMs
1. **[Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)**: Llama3-8B is released by Meta AI in April 2024.
   - Due to memory limitations, TabFSBench only provides json files for LLM fine-tuning and testing ( `datasetname_train.json / datasetname_test_i.json` , i means the degree of feature shifts), asking users to use LLM locally.
   - TabFSBench provides the context of **Credit** Dataset. Users can rewrite background, features_information, declaration and question of `llm()` in `./model/utils.py`.

#### Tabular LLMs
1. **[TabLLM](https://arxiv.org/abs/2210.10723)**: A framework that leverages LLMs for efficient tabular data classification.
2. **[UniPredict](https://arxiv.org/abs/2310.03266)**: A framework that firstly trains on multiple datasets to acquire a rich repository of prior knowledge. UniPredict-Light model that TabFSBench used is available at [Google Drive](https://drive.google.com/file/d/1ABsv0C9HSJ9-M3kpkGRIFEw-4ebKdA3h/view?usp=sharing). After downloading the model, place it in `./model/tabularLLM/files/unified/models` and rename it to `light_state.pt`.

### How to Add New Models

TabFSBench provides two methods to evaluate new model on feature-shift experiments.

1. Export the dataset. Set export_dataset as True, then can get a csv file of a given dataset in a specific experiment.
2. Import model python file.
   - Add the model name in `./run_experiment.py`.
   - Add the model function in the `./model/utils.py` by leveraging parameters like dataset, model, train_set and test_sets.
   
## Experimental Results
#### 1. Most models have the limited applicability in feature-shift scenarios.
<img src="https://s2.loli.net/2025/01/31/wvLWCdt3HrXMagG.png"  width="1000px">

#### 2. Shifted features’ importance has a linear trend with model performance degradation.
<img src="https://s2.loli.net/2025/01/31/7Hi8fX61DbTeq5L.png"  width="1000px">

We use $\Delta$ (described in equation~\ref{delta_equation}) to measure the model performance Gap $Delta$. Sum of shifted feature set's correlations refers to the sum of Pearson correlation coefficients of shifted features. Notably, model performance Gap $Delta$ and sum of shifted feature set's correlations demonstrate a strong correlation, with a Pearson correlation coefficient of $\rho$ = 0.7405.

#### 3. Model closed-environment performance correlates with feature-shift performance.
<img src="https://s2.loli.net/2025/01/31/SId5jgqNUvJxKzk.png"  width="1000px">
Model closed-environment performance vs. model feature-shift performance. Closed-environment means that the dataset does not have any degree of feature shift. Feature-shift means average model performance in all degrees of feature shifts.
