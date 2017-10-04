import tensorflow as tf
import FIFOFiles

example,label=FIFOFiles.features['i'],FIFOFiles.features['j']
batch_size=3
capacity=1000+3*batch_size
example_batch,label_batch=tf.train.batch(
    [example,label],batch_size=batch_size,capacity=capacity
)
with tf.Session() as s:
    s.run(tf.global_variables_initializer())
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord,sess=s)
    for i in range(2):
        cur_example_batch,cur_label_batch=s.run(
            [example_batch,label_batch]
        )
        print(cur_example_batch,cur_label_batch)
    coord.request_stop()
    coord.join(threads)