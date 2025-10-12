# DDHRNet
Code and dataset for paper [DDHRNet: A dual-stream high resolution network: Deep fusion of GF-2 and GF-3 data for land cover classification](https://www.sciencedirect.com/science/article/pii/S156984322200098X).

<img width="1149" height="722" alt="image" src="https://github.com/user-attachments/assets/e66dfdff-ada2-4b2d-9404-9bf29e1fb96b" />


## Latest
- Release of the Xian, Pohang and Shandong Large-Resolution Datasets, please see [Baidu Drive](https://pan.baidu.com/s/11VmCZukS2h7oc0_F11bFiA?pwd=k5wt) for details.

---

## Code
Project is based on PaddlePaddle1.8.5 (The dygraph version will be upload soon)

The Dual_stream Deep High-resolution Net (DDHRNet) model is avaliable in DDHRNet_code/models/modeling/ddhrnet.py

The discription of the dataset will be found in our paper.

Data is avaliable: [Google Drive](https://drive.google.com/file/d/1DAojDL2IjuJjW5fJLFCxj0cPqNOdHRzI/view?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/16-wNSiho5_x_Oh8g_0109w?pwd=n94h)(extract code：n94h)

Use DDHRNet_code/tools/create_data_list.py to create data list

run DDHRNet_code/train.sh for the training process

More detiles will be added

Thanks to [PaddlePaddle/PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v0.8.0)

## Citation
If you use DDHR-dataset in your research, please cite the following paper:
```
@article{ren2022dual,
  title={A dual-stream high resolution network: Deep fusion of GF-2 and GF-3 data for land cover classification},
  author={Ren, Bo and Ma, Shibin and Hou, Biao and Hong, Danfeng and Chanussot, Jocelyn and Wang, Jianlong and Jiao, Licheng},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={112},
  pages={102896},
  year={2022},
  publisher={Elsevier}
}
```
