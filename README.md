## Requirements
Our experiments are conducted under the following environments:

- Python == 3.8
- Pytorch == 1.6.0
- torchvision == 0.7.0
## Usage
#### 1. Dataset
We train the model with the OULU-NPU dataset. We need to extract frame images for  OULU-NPU video data sets. Then, we use [MTCNN](https://github.com/ipazc/mtcnn) for face detection and [PRNet](https://github.com/YadiraF/PRNet) for face depth map prediction. 
#### 2. Train the model
Train models :
```bash
python train_DC_CDN_DMU_NEW.py -lr 0.0001 -kl_lambda 0.001 -ratio 1
```
The trained model is released in ./checkpoint/
The training log is released in ./log/
 
#### 3. Test the model
Test on dataset OULU-NPU Protocol-1.
```bash
python test_DC_CDN_DMU.py
```
## Analysis

```bash lr =0.0001``` gives good results.
When lr is higher, the convergence speed is fast at first, but it will lead to failure to converge to the minimum value.
When lr is smaller, it will lead to slow convergence.





 
