# scale_estimation
This research project was intended to train vision models to have "common sense" about real-world objects' absolute sizes, without relying on contextual clues like other nearby objects. Our approach was to train vision encoders to predict the object's length/width/height from a single RGB image. Perfect accuracy is impossible since images don't provide absolute scale information, but humans nevertheless have a good intuition about objects that we hoped to replicate. Most approaches are restricted to a fixed set of hand-crafted object categories (eg. "chairs", "cars", etc.).

We train ResNets on a broad set of e-commerce data from the Amazon Berkeley Objects (ABO) dataset, exposing our model to a very diverse set of objects. Our use case treats ABO as a "crowd-sourced" dataset of image-size pairings accumulated by Amazon vendors. We initially tried direct regression, but found that classification into 10 centimeter intervals was easier. The object in the images used during evaluation may be unseen during training, indicating some degree of generalization. 

The model reached ~55% accuracy in classifying widths/heights/lengths into 10cm buckets. The model was still improving before training stopped, so there's likely still room for improvement. Below are some results on the validation set. The three numbers in the tuples are the (short, middle, longest) dimensions of each object in 10's of centimeters. So (1,2,4) means the object's dimensions falls within the (10-20cm, 20-30cm, 40-50cm) intervals.

![Sample1](./samples/1.png)
![Sample2](./samples/2.png)
![Sample3](./samples/3.png)
![Sample4](./samples/4.png)
![Sample5](./samples/5.png)
