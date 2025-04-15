## Note

 - 默认 load image 的时候没有做 normalize，因为 normalize 会导致匹配效果变差
 - 使用 ELoFTR full

## task

1. image-matching-toolbox，rubik有的都要有，除了vggt，按照之前dataset的格式
2. rubik roma数据有点奇怪，测试一下

# TODO

## Detector-based

 - [x] ALIKED+LightGlue
 - [x] DISK+LightGlue
 - [x] SP+LightGlue
 - [x] SIFT+LightGlue
 - [x] DeDoDe v2
 - [x] XFeat
 - [x] XFeat*
 - [x] XFeat+LighterGlue

 - ALIKED: https://github.com/Shiaoming/ALIKED
 - LightGlue: https://github.com/cvg/LightGlue
 - DISK: https://github.com/cvlab-epfl/disk
 - SIFT: ...
 - DeDoDe v2: https://github.com/Parskatt/DeDoDe
 - XFeat: https://github.com/verlab/accelerated_features <- LighterGlue Here
 - SP / SuperPoint: done

## Detector-free

 - [x] LoFTR - Verified
 - [x] ASpanFormer - Verified
 - [x] ELoFTR - Verified
 - [ ] RoMa (Stuck)
 - [ ] DUSt3R
 - [ ] MASt3R

 - ELoFTR: https://github.com/zju3dv/EfficientLoFTR
 - RoMa: https://github.com/Parskatt/RoMa
 - DUSt3R: https://github.com/naver/dust3r
 - MASt3R: https://github.com/naver/mast3r
 - ASpanFormer: done
 - LoFTR: done
