# MNIST-Application
I have created a very simple application that allows users to upload handwritten digits and for my model to predict them. My GitHub is very simple to use organized in different folders. The templates section contains all my pages of my website (HTML). My app.py consists of my Flask program that allows a user to upload and my model to predict a value. I additionally have the uploads section, which saves all the images the user uploads in my application. Finally, I have the static folder, which contains all my CSS files. 

## Packages Needed
- Flask v. 3.1.0
- Numpy and OS
- PyTorch v. 2.0
- Matplotlib v. 3.1.0
- Seaborn (install Python 3.8)
- Scikit Learn v. 1.6.1
- PIL v. 1.1

## Instructions 
Download all the necessary files to your local computer to have access to both the HTML and the Python code. Install Flask by using either a virtual environment or by installing Flask locally on your personal computer, whichever one is easier for you. Afterwards, make sure you install the necessary packages listed above (if you are using a virtual environment, most of them should be preinstalled). To check if the code is working properly, open your terminal and run "python3 app.py".

```terminal
python3 app.py
```

## Model Analysis
The three models I have trained before are a **Feedforward Neural Network**(FNN), a **Convolutional Neural Network**(CNN), and a **Deep Convolutional Neural Network**(DCNN). 

### FNN Model
My FNN model had three fully connected layers, which first flattens the input image, and then, through the two other layers, finally outputs one of the 10 output classes (digits 0-9). However, my FNN did not have the most incredible accuracy (only 90%) compared to CNNs and DCNNs because they have a more general-purpose application and struggle with complex image datasets. 

### CNN Model
My CNN model uses two convolutional levels, two fully connected layers, and a maxpooling layer. The first convolutional layer detects the basic features of the image, the maxpooling layer downsamples the image, the second convolutional layer learns more complex patterns in the images, and the fully connected layers combine all the features and output the final predictions of the model. 

### DCNN Model
My DCNN model uses three convolutional layers, instead of the typical two layers, and uses dropout, which prevents overfitting with the model's accuracy. Compared to CNN models, DCNN models have a higher accuracy because they have a higher number of filters, which helps the model analyze more complex elements of an image, allowing the model to have an accuracy above 98%. 
