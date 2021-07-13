"""
    model training of MC-CNN
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime
from LibMccnn.model import NET
from LibMccnn.datagenerator import ImageDataGenerator
import json

#from datageneratorXY import ImageDataGenerator

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="training of MC-CNN")
parser.add_argument("-g", "--gpu", type=str, default="0", help="gpu id to use, \
                    multiple ids should be separated by commons(e.g. 0,1,2,3)")
parser.add_argument("-ps", "--patch_size", type=int, default=11, help="length for height/width of square patch")
parser.add_argument("-bs", "--batch_size", type=int, default=512, help="mini-batch size") #should be less than image size?
parser.add_argument("-mr", "--margin", type=float, default=0.2, help="margin in hinge loss")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.002, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument("-bt", "--beta", type=int, default=0.9, help="momentum")
parser.add_argument("--print_freq", type=int, default=10, help="summary info(for tensorboard) writing frequency(of batches)")
parser.add_argument("--save_freq", type=int, default=1, help="checkpoint saving freqency(of epoches)")
parser.add_argument("--val_freq", type=int, default=1, help="model validation frequency(of epoches)")

parser.add_argument("--start_epoch", type=int, default=0, help="start epoch for training(inclusive)")

parser.add_argument("--end_epoch", type=int, default=1000, help="end epoch for training(exclusive)")

parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from. \
                    if None(default), model is initialized using default methods, if = ../data/checkpoint(mbF), model is  initialized using middleberryfast methods")
#parser.add_argument("--train_file", type=str, required=True, help="path to file containing training  \
#                    left_image_list_file s, should be list_dir/train.txt(val.txt)")
#parser.add_argument("--val_file", type=str, required=True, help="path to file containing validation \
#                    left_image_list_file s, should be list_dir/train.txt(val.txt)")
#parser.add_argument("--dataset",type=str,required=True,help="indicates the trainind data (mb or eo)")
parser.add_argument('-m','--model_foldername',type=str,required=True, help='Output folder where the models will be stored')
parser.add_argument('-d',"--dataset_foldername",type=str,required=True,help="folder path where the dataset is stored")

def test_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        
def update_filenames (train_file, dataset_foldername):
    with open(train_file, 'r') as reader:
        lines = reader.read().splitlines()
        reader.close()
    filenames = []   
    for line in lines:
        # Derive the new filename
        filenames.append(os.path.join(dataset_foldername, line))
        
    return filenames

def main():
    args = parser.parse_args()
    tf.compat.v1.disable_eager_execution()

    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    ######################
    # directory preparation
    filewriter_path = os.path.join(args.model_foldername, 'tensorboard') 
    checkpoint_path = os.path.join(args.model_foldername, 'checkpoint')  
     
    test_mkdir(filewriter_path)
    test_mkdir(checkpoint_path)
    
    ######################
    # model graph preparation
    patch_height = args.patch_size
    patch_width = args.patch_size
    batch_size = args.batch_size
    
    # Load the data from the json file
    with open('md_list.json') as f:
        data = json.load(f)

    # Add the dataset directory to the folder containing the data
    train_list = [os.path.join(args.dataset_foldername, s) for s in data['train']]
    val_list   = [os.path.join(args.dataset_foldername, s) for s in data['val']]
    
    # Data preparation
    print('Preparing Training data ...')
    train_generator = ImageDataGenerator(train_list, 1, shuffle = True,patch_size=(patch_height,patch_width))
    print('Preparing Validation data ...')
    val_generator = ImageDataGenerator(val_list, 1, shuffle = False,patch_size=(patch_height,patch_width))
    
    # Derive the number of patches per epoch
    train_batches_per_epoch = train_generator.data_size
    val_batches_per_epoch = val_generator.data_size
    
    # TF placeholder for graph input
    leftx = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, patch_height, patch_width,1])
    rightx_pos = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, patch_height, patch_width,1])
    rightx_neg = tf.compat.v1.placeholder(tf.float32, shape=[batch_size, patch_height, patch_width,1])
    
    
    # Initialize model
    left_model = NET(leftx, input_patch_size=patch_height, batch_size=batch_size)
    right_model_pos = NET(rightx_pos, input_patch_size=patch_height, batch_size=batch_size)
    right_model_neg = NET(rightx_neg, input_patch_size=patch_height, batch_size=batch_size)
    
    featuresl = tf.squeeze(left_model.features, [1, 2])
    featuresr_pos = tf.squeeze(right_model_pos.features, [1, 2])
    featuresr_neg = tf.squeeze(right_model_neg.features, [1, 2])
    
    
    # Op for calculating cosine distance/dot product
    with tf.name_scope("correlation"):
        cosine_pos = tf.reduce_sum(tf.multiply(featuresl, featuresr_pos), axis=-1)
        cosine_neg = tf.reduce_sum(tf.multiply(featuresl, featuresr_neg), axis=-1)

    # Op for calculating the loss
    with tf.name_scope("hinge_loss"):
        margin = tf.ones(shape=[batch_size], dtype=tf.float32) * args.margin
        loss = tf.maximum(0.0, margin - cosine_pos + cosine_neg)
        loss = tf.reduce_mean(loss)

        loss_pos = tf.reduce_mean(tf.maximum(0.0, cosine_pos))
        loss_neg = tf.reduce_mean(tf.maximum(0.0, cosine_neg))

    # Train op
    with tf.name_scope("train"):
        var_list = tf.compat.v1.trainable_variables()
        for var in var_list:
            print("{}: {}".format(var.name, var.shape))
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))
  
        # Create optimizer and apply gradient descent with momentum to the trainable variables
        optimizer = tf.compat.v1.train.MomentumOptimizer(args.learning_rate, args.beta)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # summary Ops for tensorboard visualization
    with tf.name_scope("training_metric"):
        training_summary = []
        # Add loss to summary
        training_summary.append(tf.compat.v1.summary.scalar('hinge_loss', loss))

        # Merge all summaries together
        training_merged_summary = tf.compat.v1.summary.merge(training_summary)

    # validation loss
    with tf.name_scope("val_metric"):
        val_summary = []
        val_loss = tf.compat.v1.placeholder(tf.float32, [])

        # Add val loss to summary
        val_summary.append(tf.compat.v1.summary.scalar('val_hinge_loss', val_loss))
        val_merged_summary = tf.compat.v1.summary.merge(val_summary)

    # Initialize the FileWriter
    writer = tf.compat.v1.summary.FileWriter(filewriter_path)
    # Initialize an saver for store model checkpoints
    saver = tf.compat.v1.train.Saver(max_to_keep=10)

    ######################
    # DO training 
    # Start Tensorflow session
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                        log_device_placement=False, \
                        allow_soft_placement=True, \
                        gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))) as sess:
     
        # Initialize all variables
        sess.run(tf.compat.v1.global_variables_initializer())

        # resume from checkpoint or not
        if args.resume is None:
            # Add the model graph to TensorBoard before initial training
            writer.add_graph(sess.graph)
        else:
            #saver.restore(sess, args.resume)
            checkpoint = args.resume
            ckpt = tf.train.get_checkpoint_state(checkpoint)
            print("{}: restoring from {}...".format(datetime.now(), checkpoint))
            print("{}: restoring from {}...".format(datetime.now(), ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

        print("training_batches_per_epoch: {}, val_batches_per_epoch: {}.".format(\
                train_batches_per_epoch, val_batches_per_epoch))
        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(), 
                                                        filewriter_path))
      
        # Loop training
        min_val_loss = 0.2 # 0.12278874218463899
        loss_flag = False

        for epoch in range(args.start_epoch, args.end_epoch):
            print("{} Epoch number: {}".format(datetime.now(), epoch+1))

            for batch in tqdm(range(train_batches_per_epoch)):
                # Get a batch of data
                #if args.nchann == 1:
                batch_left, batch_right_pos, batch_right_neg = train_generator.next_batch(batch_size)
                #elif args.nchann == 2:
                #    batch_left, batch_right_pos, batch_right_neg = train_generator.next_batch_2CH(batch_size)
                
                # And run the training op
                sess.run(train_op, feed_dict={leftx: batch_left,
                                              rightx_pos: batch_right_pos,
                                              rightx_neg: batch_right_neg})

                # Generate summary with the current batch of data and write to file
                if (batch+1) % args.print_freq == 0:
                    s = sess.run(training_merged_summary, feed_dict={leftx: batch_left,
                                                                     rightx_pos: batch_right_pos,
                                                                     rightx_neg: batch_right_neg})
                    writer.add_summary(s, epoch*train_batches_per_epoch + batch)

                
            if (epoch+1) % args.val_freq == 0:
                # Validate the model on the entire validation set
                print("{} Start validation".format(datetime.now()))
                val_ls = 0.
                val_ls_pos = 0.
                val_ls_neg = 0.

                for _ in tqdm(range(val_batches_per_epoch)):
                    #if args.nchann == 1:
                    batch_left, batch_right_pos, batch_right_neg = val_generator.next_batch(batch_size)
                    #elif args.nchann == 2:
                    #    batch_left, batch_right_pos, batch_right_neg = val_generator.next_batch_2CH(batch_size)
                    if loss_flag:
                        result_pos = sess.run(loss_pos, feed_dict={leftx: batch_left,
                                                         rightx_pos: batch_right_pos,
                                                         rightx_neg: batch_right_neg})
                        result_neg = sess.run(loss_neg, feed_dict={leftx: batch_left,
                                                         rightx_pos: batch_right_pos,
                                                         rightx_neg: batch_right_neg})

                    result = sess.run(loss, feed_dict={leftx: batch_left,
                                                         rightx_pos: batch_right_pos,
                                                         rightx_neg: batch_right_neg})
                    val_ls += result
                    if loss_flag:
                        val_ls_pos += result_pos
                        val_ls_neg += result_neg


                val_ls = val_ls / (1. * val_batches_per_epoch)
                val_ls_pos = val_ls_pos/ (1. * val_batches_per_epoch)
                val_ls_neg = val_ls_neg/ (1. * val_batches_per_epoch)
                
                print('validation loss: {}'.format(val_ls))
                if loss_flag:
                    print('loss_pos:{}\nloss_neg:{}'.format(val_ls_pos,val_ls_neg))
                s = sess.run(val_merged_summary, feed_dict={val_loss: np.float32(val_ls)})
                writer.add_summary(s, train_batches_per_epoch*(epoch + 1))

            if (epoch+1) % args.save_freq == 0:
                if min_val_loss > val_ls:
                    min_val_loss = val_ls
                    print("{} Saving checkpoint of model...".format(datetime.now()))
                    # save checkpoint of the model
                    checkpoint_name = os.path.join(checkpoint_path, 'model_epoch'+str(epoch+1)+'.ckpt')
                    save_path = saver.save(sess, checkpoint_name)

            # Reset the file pointer of the image data generator
            val_generator.reset_pointer()
            train_generator.reset_pointer()

if __name__ == "__main__":
    main()
        
