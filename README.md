# PFLD_multitask_by_tf
multitask network based on PFLD

## fix errors
 if there is any error in the codes , just remove the line in which the errors are
## load data
refer to gen_data.py
## dadaset and annotation
as for  my dataset and annotations  
I have four kinds of dataset and for every dataset there is a especial indicates_str in gen_data.py
every line in the indicate_str represent an attribute and the first follwing number indicates whether the attribute is recorded in the annotation file . The second following number indicates how many values the attribute has or how long the array representing the attributs is.  
For example, if datatype == 'imdb' ,then  
indicates_str = '''lmk 0 212  
                race_gt 0 4  
                race_st 1 1  
                angles_by_gt 0 3  
                angles_by_st 1 3  
                expressions 1 10  
                attr6_gt 0 5  
                attr6_st 1 5  
                age_gt 1 1  
                age_st 1 1  
                gender_gt 1 1  
                gender_st 1 1  
                young 0 1  
                mask 1 1  
                open_eye_gt 0 1  
                open_eye_st 1 1  
                mouth_open_slightly 0 1  
                mouth_open_widely 0 1  
                sunglasses_gt 0 1  
                sunglasses_st 1 1  
                forty_attr 0 33'''  
  That is :  
      the imdb dataset does not have attributes lmk, race_gt , angles_by_gt or etc.  
      But it has race_st, angles_by_st, expressions and etc.  
      We can infer by the numbers in the third col that lmk should be an array with length of 212 and race_gt should be an array with lenth of 4 and so on.  
## train
refer to train_model_v4_dynamic_weights.py and train-scripts/train-v4-dynamic_weights.sh
change the parameters in train-scripts/train-v4-dynamic_weights.sh as you need and run this script


