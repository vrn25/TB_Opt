# TB_Opt: A General Game Theoretic Recursive Framework for Turn-Based Optimization

This repository contains the PyTorch code for TB_Opt framework. The project report can be found [here](https://drive.google.com/file/d/1ujJ0_OtIe10XNCQf9StGYnVAQCqv8MXH/view?usp=sharing). 

The code is organized as follows:
```
.
├── README.md
├── x_y_x2y2
│   ├── optimizers_generic.py
│   ├── plots
│   │   ├── Bilinear_Bilinear
│   │   ├── Bilinear_Linear
│   │   ├── Linear_Bilinear
│   │   └── Linear_Linear
│   └── run.py   
├── x_y_xy
    ├── optimizers_xy.py
    ├── plots
    │   ├── Bilinear_Bilinear
    │   ├── Bilinear_Linear
    │   ├── Linear_Bilinear
    │   └── Linear_Linear
    └── run.py
```
[`x_y_xy`](https://github.com/vrn25/TB_Opt/tree/main/x_y_xy) contains the code for the game  ![equation](https://latex.codecogs.com/gif.latex?\max_x&space;\min_y&space;x&plus;y&plus;xy) and [`x_y_x2y2`](https://github.com/vrn25/TB_Opt/tree/main/x_y_x2y2) contains the code for the game  ![equation](https://latex.codecogs.com/gif.latex?\max_x&space;\min_y&space;x&plus;y&plus;x^2y^2). 

[`optimizers_generic`](https://github.com/vrn25/TB_Opt/blob/main/x_y_x2y2/optimizers_generic.py) contains the the code for a generic optimizer (_linear-linear, linear-bilinear, bilinear-linear, bilinear-bilinear_ cases) that works for any objectives ![equation](https://latex.codecogs.com/gif.latex?\eta^1) and ![equation](https://latex.codecogs.com/gif.latex?\eta^2). Further details can be found [here](https://drive.google.com/file/d/1ujJ0_OtIe10XNCQf9StGYnVAQCqv8MXH/view?usp=sharing). 
