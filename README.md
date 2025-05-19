# AerialExtreMatch: A Benchmark for Extreme-View Image Matching and Localization
### [Project Page](https://xecades.github.io/AerialExtreMatch/) | Paper (WIP)

<br />

> **AerialExtreMatch: A Benchmark for Extreme-View Image Matching and Localization**  
> [Rouwan Wu<sup>1</sup>](https://github.com/RingoWRW), [Zhe Huang<sup>2</sup>](https://github.com/Xecades), [Xingyi He<sup>2</sup>](https://hxy-123.github.io/), [Yan Liu<sup>3</sup>](https://faculty.hdu.edu.cn/jsjxy/ly2_21682/main.htm), [Shen Yan<sup>1</sup>](https://openreview.net/profile?id=~Shen_Yan6), [Sida Peng<sup>2</sup>](https://pengsida.net/), [Maojun Zhang<sup>1&dagger;</sup>](https://orcid.org/0000-0001-6748-0545), [Xiaowei Zhou<sup>2&dagger;</sup>](https://xzhou.me/)  
> National University of Defense Technology<sup>1</sup>, State Key Lab of CAD&CG, Zhejiang University<sup>2</sup>, Huazhong University of Science and Technology<sup>3</sup>  
> <!-- NeurIPS --> 2025

<p align="center">
    <img src="assets/teaser.png" alt="teaser" width=100%>
    <br>
    <em>We introduce <b>AerialExtreMatch</b>, a large-scale, high-fidelity benchmark tailored for extreme-view image matching and UAV localization. It consists of three datasets: <b>Train Pair</b>, <b>Evaluation Pair</b>, and <b>Localization</b>. All code and datasets are readily available for public access.</em>
</p>

> [!WARNING]  
> Docs are under preparation, and will be released soon.

## Resources

> [!IMPORTANT]  
> In our paper, TWO seperate codebases are provided: **benchmarking** and code of our pretrained **RoMa** model.  
> To increase simplicity and consistency, we slightly abuse the concept of git branches and **make the two codebases as branches of this repository**.

 - **Code**
   - [**`Benchmark` branch**](https://github.com/Xecades/AerialExtreMatch/tree/Benchmark): source code for the benchmark, including feature matching and localization pipelines for models mentioned in the paper. **(WIP)**
   - [**`RoMa` branch**](https://github.com/Xecades/AerialExtreMatch/tree/RoMa): the code we use to train our RoMa model.
 - **Dataset**
   - [**AerialExtreMatch-Train**](https://huggingface.co/datasets/Xecades/AerialExtreMatch-Train): corresponds to **Train Pair** set.
   - [**AerialExtreMatch-Benchmark**](https://huggingface.co/datasets/Xecades/AerialExtreMatch-Benchmark): corresponds to **Evaluation Pair** set.
   - [**AerialExtreMatch-Localization**](https://huggingface.co/datasets/Xecades/AerialExtreMatch-Localization): corresponds to **Localization** set.
 - **Checkpoints**: see [[Release]](releases).

## Introduction

WIP.

## License

[MIT License](LICENSE)
