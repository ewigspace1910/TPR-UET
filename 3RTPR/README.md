## Introduction
PyTorch implementation for [ Enhancing Text-Based Person Retrieval by Combining Fused Representation and Reciprocal Learning with Adaptive Loss Refinement]() . 

### News!

- [07/2025] Push Initial rep on Github. Citation, official code will be updated when the paper is published.


### 3RTPR framework
```
Text-based person retrieval is defined as the challenging task of searching for people's images based on given textual queries in natural language. Conventional methods primarily use deep neural networks to understand the relationship between visual and textual data, creating a shared feature space for cross-modal matching. The absence of awareness regarding variations in feature granularity between the two modalities, coupled with the diverse poses and viewing angles of images corresponding to the same individual, may lead to overlooking significant differences within each modality and across modalities, despite notable enhancements. Furthermore, the inconsistency in caption queries in large public datasets presents an additional obstacle to cross-modality mapping learning. Therefore, we introduce 3RTPR, a novel text-based person retrieval method that integrates a representation fusing mechanism and an adaptive loss refinement algorithm into a dual-encoder branch architecture. Moreover, we propose training two independent models simultaneously, which reciprocally support each other to enhance learning effectiveness. Consequently, our approach encompasses three significant contributions: (i) proposing a fused representation method to generate more discriminative representations for images and captions; (ii) introducing a novel algorithm to adjust loss and prioritize samples that contain valuable information; and (iii) proposing reciprocal learning involving a pair of independent models, which allows us to enhance general retrieval performance. In order to validate our method's effectiveness, we also demonstrate superior performance over state-of-the-art methods by performing rigorous experiments on three well-known benchmarks: CUHK-PEDES, ICFG-PEDES, and RSTPReid. 

```



## Requirements and Datasets
- Same as [IRRA](https://github.com/anosorae/IRRA)


## Training and Evaluation

### Training new models

```json

```

### Evaluation
Please check the predefined script in ```test.sh``` file and run it.

 

### Experiment Results:
 - will be updated after paper is accepted


## Citation


## Acknowledgements
The code is based on [IRRA](https://github.com/anosorae/IRRA) licensed under Apache 2.0.
