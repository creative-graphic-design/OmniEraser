# 🗝 OmniEraser

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://pris-cv.github.io/Omnieraser/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.07397-b31b1b.svg)](https://arxiv.org/pdf/2501.07397)
[![Hugging Face](https://img.shields.io/badge/Demo-🤗%20Hugging%20Face-blue)](https://huggingface.co/spaces/theSure/Omnieraser)

<p align = "center">
<img src="https://raw.githubusercontent.com/PRIS-CV/Omnieraser/refs/heads/main/static/images/logo_transparent.png" width="150" />
</p>

This is the code release for paper: "OmniEraser: Remove Objects and Their Effects in Images with Paired Video-Frame Data".

**Abstract**: Inpainting algorithms have achieved remarkable progress in removing objects from images, yet still face two challenges: 1) struggle to handle the object's visual effects such as shadow and reflection; 2) easily generate shape-like artifacts and unintended content.In this paper, we propose `Video4Removal`, a large-scale dataset comprising over 100,000 high-quality samples with realistic object shadows and reflections. By constructing object-background pairs from video frames with off-the-shelf vision models,the labor costs of data acquisition can be significantly reduced.To avoid generating shape-like artifacts and unintended content, we propose Object-Background Guidance, an elaborated paradigm that takes both the foreground object and background images.It can guide the diffusion process to harness richer contextual information.Based on the above two designs, we present `OmniEraser`, a novel method that seamlessly removes objects and their visual effects using only object masks as input. Extensive experiments show that OmniEraser significantly outperforms previous methods, particularly in complex in-the-wild scenes. And it also exhibits a strong generalization ability in anime-style images.

## News

- **Apr 18 2025**: 📂 We have open-sourced our [RemovalBench](https://huggingface.co/datasets/BaiLing/RemovalBench/tree/main), which we use to evaluate the model as well as our evaluation results.
- **Apr 12 2025**: 🎯 We have released the [Omnieraser Model weight (ControlNet version)](https://huggingface.co/theSure/Omnieraser_Controlnet_version/tree/main), which enables more accurate object removal with significantly better background consistency, thanks to the assistance of [alimama ControlNet](https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta). For this weight we consider only the purest object removal, you can find the code and the gradio demo in the [ControlNet_version folder](https://github.com/PRIS-CV/Omnieraser/tree/main/ControlNet_version).
- **Apr 12 2025**: 🖥️ Omnieraser Inference and the offline gradio script (Base & ControlNet version) are released!
- **Apr 08 2025**: 🤗 We have released our fantastic model demo (Base version) at [HuggingFace Spaces](https://huggingface.co/spaces/theSure/Omnieraser), have fun with it!! (Unfortunatey, you cannot sketch mask here due to the compatibility issues with zerogpu, if you need draw mask manually, please use our offline gradio script.)
- **Apr 06 2025**: 🎯 We have released the [Omnieraser Model weight (Base version)](https://huggingface.co/theSure/Omnieraser/tree/main)!
- **Apr 02 2025**: 🖥️ Omnieraser Training code [(ControlNet version)](https://github.com/PRIS-CV/Omnieraser/tree/main/ControlNet_version) is released!
- **Apr 01 2025**: 🖥️ Omnieraser Training code (Base version) is released!
- **Mar 15 2025**: 🔥 Our [Project Page](https://pris-cv.github.io/Omnieraser/) has been published!

|     | TODO Task     | Update                   |
| --- | ------------- | ------------------------ |
| ☐   | 📂 **Dataset** | Under active preparation |

## Citation

If you find this code repository useful in your research, please consider citing our paper:

```
@article{wei2025omnieraserremoveobjectseffects,
title={OmniEraser: Remove Objects and Their Effects in Images with Paired Video-Frame Data},
author={Runpu Wei and Zijin Yin and Shuo Zhang and Lanxiang Zhou and Xueyi Wang and Chao Ban and Tianwei Cao and Hao Sun and Zhongjiang He and Kongming Liang and Zhanyu Ma},
journal={arXiv preprint arXiv:2501.07397},
year={2025},
url={https://arxiv.org/abs/2501.07397},
}
```
