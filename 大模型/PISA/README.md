# PromptCast: A New Forecasting Paradigm

## Introduction
This repository is the reporisity of **PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting** (currently under submission). PISA is a large-scale dataset including three real-world forecasting scenarios (three sub-sets) with 311,932 data instances in total. It is designed to support and facilitate the novel PromptCast task proposed in the paper. 


## Numerical Time Series Forecasting vs. PromptCast
![](resources/concept.png)

> Exisiting numerical-based forecasting VS. Prompt-based forecasting

### ***PromptCast Evaluation Metrics***
- RMSE
- MAE
- Missing Rate: whether the numerical forecasting target can be decoded (via string parsing) from the generated output prompts.


## PISA Dataset
### ***Forecasting Scenarios***
The proposed PISA dataset contrains three real-world forecasting scenarios:
- CT: city temperature forecasting
- ECL: electricity consumption forecasting
- SG: humana mobility visitor flow forecasting

![](resources/statistics.png)

> Details of three sub-sets
> 
<br></br>

### ***Folder Structure (see [Dataset](Dataset/README.md))***
```text
Dataset
|── PISA-Prompt
    │── CT
        │-- train_x_prompt.txt
        │-- train_y_prompt.txt
        │-- val_x_prompt.txt
        │-- val_y_prompt.txt
        │-- test_x_prompt.txt
        │-- test_y_prompt.txt
    │── ECL
        │-- train_x_prompt.txt
        │-- train_y_prompt.txt
        │-- val_x_prompt.txt
        │-- val_y_prompt.txt
        │-- test_x_prompt.txt
        │-- test_y_prompt.txt  
    │── SG
        │-- train_x_prompt.txt
        │-- train_y_prompt.txt
        │-- val_x_prompt.txt
        │-- val_y_prompt.txt
        │-- test_x_prompt.txt
        │-- test_y_prompt.txt   
```

## Benchmark Results
Please check [Benchmark](Benchmark/README.md) folder for the implementations of benchmarked methods.
<br></br>

![](resources/result_1.png)

> RMSE and MAE performance
> 
<br></br>

![](resources/result_2.png)

> Missing Rate results
> 
<br></br>

![](resources/result_3.png)

> Results under train-from-scratch and cross-scenario zero-shot settings
> 
<br></br>

If you think this repo is useful, please cite our papers



@article{xue2022prompt,
  title={Prompt-Based Time Series Forecasting: A New Task and Dataset},
  author={Xue, Hao and Salim, Flora D},
  journal={arXiv preprint arXiv:2210.08964},
  year={2022}
}

@inproceedings{xue2022translating, 
  title={Translating human mobility forecasting through natural language generation}, 
  author={Xue, Hao and Salim, Flora D and Ren, Yongli and Clarke, Charles LA}, 
  booktitle={Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining}, 
  pages={1224--1233}, 
  year={2022} 
} 

@inproceedings{xue2022leveraging, 
  title={Leveraging language foundation models for human mobility forecasting}, 
  author={Xue, Hao and Voutharoja, Bhanu Prakash and Salim, Flora D}, 
  booktitle={Proceedings of the 30th International Conference on Advances in Geographic Information Systems}, 
  pages={1--9}, 
  year={2022} 
} 

