from utils import loading_data, text_cleaning, concatenation

training,validation=loading_data("../data/twitter_training.csv","../data/twitter_validation.csv")
training=text_cleaning(training) #(73996, 2) shape
validation=text_cleaning(validation) #(1000, 2) shape
print(training.shape)
print(validation.shape)
training_data,testing_data=concatenation(training,validation)
print(training_data.shape) #(56247, 2) shape
print(testing_data.shape)#(18749, 2) shape

#working on tokenizer