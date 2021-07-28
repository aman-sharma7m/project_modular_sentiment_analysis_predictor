# project_modular_sentiment_analysis_predictor
It is a sentiment analysis predictor with the structural format.

Desc:
This model is based on a structural format to deploy prediction model based on the 
data we feed.userid & project id needs to be passed while training the model it will make
directory as per userid->projectId. Thereafter it cleans the data and prepare the vectorizer and 
naive bayes algo-> multinomialNB for that specific projectId .

When it oomes to test the predictor, we again require userId & projectId with data to test the suitable model.

Input data format can be seen in :
user_test_data_send
user_train_data_send

Application:
It can be easily used to review analysis for different departments in a organisation.
each dept will have their own review data + model.
By that sentiment can be analyze over dept's review.

Future approach:
Thinking of collaborating it with reviewscrapper that 
i made, with that it would be easy for me to design review sentiment analysis
for each website(flipkart,amazon) and for each product seperate predictor.
