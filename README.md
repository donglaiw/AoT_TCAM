# AoT_TCAM
[[project page]](http://aot.csail.mit.edu/)
[[dataset and pre-process tools]](https://github.com/donglaiw/AoT_Dataset)

Torch model for arrow of time prediction in the CVPR 18 paper

D. Wei, J. Lim, A. Zisserman, W. Freeman.
<b>"Learning and Using the Arrow of Time."</b>
in CVPR 2018. 

## Demo:
1. training from scratch flow-TCAM model for AoT prediction  on UCF101
```
CUDA_ID=0,1,2;GPU_ID=1,2,3;CPU_N=10;
N_BATCH=32;N_FRAME=20;N_C=2
M_LABEL=2;M_LOSS=0;M_BEND=cudnn
M_DROPOUT=0.5,0.5;M_ARCH=vggbn_tcam_pair2
D_TYPE=2;D_PRE=5;D_NAME=flow%s_%04d.jpg;D_CROP=2;D_TCROP=1;
D_OUT=results/ucf_train/
D_TXT=data/@01_cnn_ta_flow_orig_fb.txt
E_SAVE=5;E_ALL=20;E_ST=1;E_SIZE=5000;E_ITER=3
E_PARAM=5.1

CUDA_VISIBLE_DEVICES=${CUDA_ID} th main_video.lua -GPUs ${GPU_ID} -nDonkeys ${CPU_N} \
    -batchSize ${N_BATCH} -vnF ${N_FRAME} -vnC ${N_C} \
    -lossType ${M_LOSS} -nClasses ${M_LABEL}  -Mdropout ${M_DROPOUT} -netType ${M_ARCH}  -backend ${M_BEND} \
    -cropType ${D_CROP} -TcropType ${D_TCROP} -cache ${D_OUT} -data ${D_TXT} -xType ${D_PRE} -imType ${D_TYPE} -imFormat ${D_NAME} \
    -retrain train -retrainOpt train -paramId ${E_PARAM} \
    -nEpochs ${E_ALL} -epochNumber ${E_ST} -epochSave ${E_SAVE} -epochSize ${E_SIZE} -iter_size ${E_ITER} 2>&1 | tee ${Dout}/log-${N_BATCH}-${N_FRAME}-${N_C}-${M_DROPOUT}-${M_ARCH}-${D_TYPE}-${D_PRE}-${D_CROP}-${D_TCROP}-${E_ALL}-${E_ST}-${E_ITER}.log
```

## Tips:
- to debug in lua, add this line:
`local dbg = require('util/debugger');dbg()`

- to visualize data in matlab, add this line:
`local matio = require 'matio'; matio.save('test.mat',{t1=data1,t2=data2})`

## Reference Torch Packages:
1.  [[multi-gpu trainining]](https://github.com/soumith/imagenet-multiGPU.torch)
2.  [[debugger]](https://github.com/slembcke/debugger.lua)
3.  [[matlab data i/o]](https://github.com/soumith/matio-ffi.torch)


## Citation
Please cite our paper if you find it useful for your work:
```
@inproceedings{wei2018learning,
  title={Learning and Using the Arrow of Time},
  author={Wei, Donglai and Lim, Joseph J and Zisserman, Andrew and Freeman, William T},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={8052--8060},
  year={2018}
}
```
