1. Set Up Azure Resources
1.	Create a Resource Group
•	Log in to Azure Portal: Go to Azure Portal and sign in with your credentials.
•	Navigate to Resource Groups: In the left-hand menu, select Resource Groups (or search for it in the search bar).
•	Create New Resource Group: Click the + Create button.
1.	Subscription: Choose the subscription.
2.	Resource Group Name: Enter a unique name.
3.	Region: Select the desired region for the group.
4.	Review and Create: Review the details, click Review + Create, then click Create to finalize.
5.	Verify: Confirm the resource group is listed under Resource Groups.

2. Create an Azure Storage Container
Azure Blob Storage is used to store application logs.
1.	Navigate to Storage Accounts in Azure Portal.
2.	Click Create.
•	Resource Group: ImageClass
•	Name: 
•	Region: West US.
•	After the storage account is created, go to Containers under the storage account.
•	Click + Container.
•	Name: logs
•	Public Access Level: Private (default).
3.	Upload the application logs to the container using Azure Portal, Azure CLI, or Azure Storage Explorer.

3.	 Set Up Azure Databricks Workspace
1.	Log in to Azure Portal: Go to Azure Portal.
2.	Navigate to Databricks: Search for and select Azure Databricks.
3.	Click + Create: Start the workspace creation process.
   
5.	Fill in Details:
•	Subscription: Student
•	Resource Group: ImageClass
•	Workspace Name: ImageCls
•	Region: West US
•	Pricing Tier: Standard
•	Review and Create: Click Review + Create, then Create.
•	Access Workspace: Once deployed, click Launch Workspace to start using Azure Databricks.

4.	Create a Databricks Cluster
1.	Access Databricks Workspace: Log in to your Azure Databricks workspace.
2.	Navigate to Clusters: In the left-hand menu, click Clusters.
3.	Create Cluster: Click + Create Cluster.
4.	Configure Cluster:
1.	Cluster Name: Akhil Mittapalli.
2.	Cluster Mode: Standard.
3.	Databricks Runtime: 10.4 LTS (includes Apache Spark 3.2.1, Scala 2.12).
4.	Autoscaling: Enable or configure autoscaling (optional).
5.	Node Types: Standard_DS3_V2
6.	Worker/Driver Count: Single node (Student version- only single node is allowed)
5.	Create: Click Create Cluster to launch.
6.	Wait for Activation: Wait for the cluster to start (status: Running).

5. Mount Azure Blob Storage in Databricks
1.	Prepare Storage Details:
2.	Storage Account Name:imageclsstorage
3.	Container Name:input
4.	Access Key: Retrieve the storage account key from Azure Portal.
5.	Access Databricks Workspace: Log in to your Azure Databricks workspace.
6.	Open a Notebook: Navigate to Workspace and create or open a notebook.
7.	Run Mount Command: Use the following code to mount the storage:
8.	Use the Mount: Access the mounted blob storage using /mnt/mycontainer.

6.  Preprocessing and Training in Databricks
1.	Import Libraries:
•	SparkSession from pyspark.sql to initialize the Spark session.
•	tensorflow as tf for deep learning functionalities.
•	ImageDataGenerator from tensorflow.keras.preprocessing.image for image augmentation.
•	Model, Sequential, Dense, Flatten, Dropout from tensorflow.keras.models and tensorflow.keras.layers for building the neural network.
•	EarlyStopping from tensorflow.keras.callbacks for early stopping during training.
•	os for operating system functionalities.

2.	Initialize Spark Session:
•	Create a Spark session named "ImageClassification".

3.	Define Paths to Datasets:
•	Specify paths to datasets stored in Databricks File System (DBFS).

4.	Define Categories:
•	List the categories for image classification.




7. Model Training and Evaluation
This section focuses on training the image classification model using the preprocessed data and evaluating its performance.
Steps:
1.	Model Architecture:
•	Define the architecture of the neural network using TensorFlow's Keras API.
•	Use layers such as Conv2D, MaxPooling2D, Flatten, Dense, and Dropout to build the model.
2.	Compile the Model:
•	Compile the model with an appropriate optimizer (e.g., Adam), loss function (e.g., categorical cross-entropy), and evaluation metrics (e.g., accuracy).
3.	Data Generators:
•	Use ImageDataGenerator for data augmentation and to create training and validation data generators.
4.	Callbacks:
•	Implement callbacks such as EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau to optimize the training process.
5.	Model Training:
•	Train the model using the fit method, specifying the training and validation data generators, number of epochs, and callbacks.
6.	Model Evaluation:
•	Evaluate the trained model on the validation dataset to assess its performance.
•	Generate and visualize metrics such as accuracy and loss.
7.	Model Saving:
•	Save the trained model for future use.


8. Visualize Training and Validation Results
This section focuses on visualizing the training and validation results to evaluate the performance of the model.
Steps:
1.	Import Libraries:
•	Import matplotlib.pyplot for plotting the results.
2.	Define Plotting Function:
•	Create a function plot_results that takes the results dictionary as input.
•	For each dataset, generate plots for training and validation accuracy and loss across epochs.
3.	Generate Plots:
•	Use the plot_results function to visualize the accuracy and loss for each dataset.
•	Display the plots to compare the training and validation metrics.

9. Model Evaluation and Testing
This section focuses on evaluating and testing the saved model to assess its performance on unseen data.
Steps:
1.	Load the Saved Model:
•	Load the best model saved during training using TensorFlow's load_model function.
2.	Preprocess Test Data:
•	Use ImageDataGenerator to preprocess the test data. This includes rescaling the pixel values to the range [0, 1].
•	Create a test data generator to load and preprocess images in batches from the specified test data directory.
3.	Evaluate the Model:
•	Evaluate the model on the test dataset to calculate the test loss and accuracy.
•	Print the test loss and accuracy to assess the model's performance.
4.	Generate Predictions:
•	Use the model to generate predictions on the test dataset.
•	Convert the predicted probabilities to class labels.
5.	Compare Predictions to True Labels:
•	Extract the true labels from the test data generator.
•	Generate a classification report to compare the predicted labels to the true labels. This report includes precision, recall, and F1-score for each class.
6.	Display Confusion Matrix:
•	Generate a confusion matrix to visualize the performance of the model.
•	Use Seaborn to create a heatmap of the confusion matrix, displaying the true labels on the y-axis and the predicted labels on the x-axis.

10.Model deployment as real time end point

1.Register the Model**:
   1.The model logged with MLflow is automatically registered in the Azure ML Model Registry.

2.Create a Real-Time Endpoint:
   1. Navigate to Azure Machine Learning Studio.
   2. Go to Models > Select the registered model.
   3. Click Deploy > Real-Time Endpoint.


11. Endpoint testing
Steps:
1. Image Data Preparation for Model Inference
This step involves defining functions to preprocess single or multiple images for inference. The images are resized to a target size, normalized, and prepared for input into a trained model.
2. Preprocess Images
•	Function preprocess_image: Preprocesses a single image by resizing it to the target dimensions, normalizing pixel values, and adding a batch dimension.
•	Function preprocess_images: Processes a list of image paths by applying the preprocess_image function to each image and returning a batch of preprocessed image arrays.
3. Model Scoring via Databricks API
This step defines a function to score a model by sending inference requests to a Databricks API endpoint. The function converts the input data into the required format, sends a POST request with authentication, and returns the prediction results from the model.
4. Create JSON Payload
•	Function create_tf_serving_json: Converts the input data into a JSON payload suitable for TensorFlow Serving.
5. Define Scoring Function
•	Function score_model: Takes a dataset as input, converts it into the required JSON format, sends a POST request to the Databricks API endpoint with the JSON payload and authentication headers, and returns the prediction results.
6. Set Up API Endpoint and Authentication
•	Specify the URL of the Databricks API endpoint.
•	Use an environment variable to securely store and access the Databricks token for authentication.
7. Handle API Response
•	Check the status code of the response.
•	Raise an exception if the request fails, otherwise return the prediction results.
8. Call the Scoring Function
•	Use the score_model function to send the preprocessed image data to the Databricks API endpoint and print the predictions.
9. The predictions with highest prediction score is mapped to a label.
