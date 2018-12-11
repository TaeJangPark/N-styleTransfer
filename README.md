# N-Style Transfer

### Installation
if you want to use pretrained models, then all you need to do is:
```sh
git clone https://github.com/djang000/N-styleTransfer.git
```

if you also want to train new modes, you will need the MS-COCO or other natural images for training files and VGG wegihts by running.

you can download VGG_16 weight by using below command
```sh
wget https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
```
To prepare the MS-Coco dataset for use with train.py, you will have to convert it to Tensorflow's TFRecords format, which shards the images into large files for more efficient reading from disk. tfrecords_writer.py can be used for this as shown below. Change --num_threads to however many threads your cores can handle, and ensure that it divides whatever you choose for --train_shards. This block will give shards ~100MB in size:
```sh
python tfrecords_writer.py --train_directory /path/to/training/data \
                           --output_directory /path/to/desired/tfrecords/location \
                           --train_shards 126 \
                           --num_threads 6 
```

To prepare your style images, you just run this file:
```sh
image_stylization_create_dataset \
      --vgg_checkpoint=/path/to/vgg16_weights.npz \
      --style_files=/path/to/style/images/*.jpg \
      --output_file=/tmp/image_stylization/style_images.tfrecord
```


### Usage

Following are examples of how the scripts in this repo can be used. Details on all available options can be viewed by typing python stylize_image.py -h into your terminal (replacing with script of interest).

- To generate stylize images with pre-trained model

```sh
python stylize_image.py --input_img=/path/to/input.jpg \
                        --output_img=/path/to/out.jpg \
                        --checkpoint=models//model_name.ckpt
                        --num_styles=3 \
                        --which_styles="[0, 1, 2]"
```	
if you want to generate each stylize images, you just set which_styles optios to list like this "which_style=[0, 1, ... num_stlys-1]".
For generating mixture image, which_styles is dict and just set this style "{0:0.2, 1:0.3, 2:0.5}".

- To train a model

Creates a trained neural net that can be used to stylize images. Tensorboard logs of the loss functions and checkpoints of the model are also created. Note that this will take a long time to get a good result. Example usage:
```sh
python train.py --train_dir=/path/to/tfrecords \
                --style_dataset=/style_images/style_images.tfrecord \
                --model_name=model_name \
                --num_styles=num_styles \
                --vgg_checkpoint=/path/to/vgg/.ckpt
```


