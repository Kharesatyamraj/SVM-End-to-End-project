# SVM-End-to-End-project
About Model: On giving required inputs model is supposed to give name of the species of flowers as output.
About Dataset : IRIS dataset contains features like Sepal length/width ,Petal length/width and species of flowers.

In assignment AI model based on SVM algorith is developed,different SVM kernels tested 
on IRIS dataset among them linear_kernel and rbf_kernel gives the best right fit models.

>Accuracy table with different kernels is given below 
   
	Kernels  Test Accuracy  Train Accuracy
0	linear       0.955556        0.960784
1	poly         1.000000        0.980392
2	rbf          0.955556        0.960784
3	sigmoid       0.466667        0.352941


We chosen rbf_kernel for python flask app development.
Created back-end with python flask liberary.
Flasgger and Swagger liberaries are used fro front-end development.
Model gives nearly 95% accuracy.

Files in Assignment Folder:

#Dockerfile
#requirements
#Assignment Details

>>> app folder: 
#IRIS.csv - dataset
#main.py - Flask app code file
#svm.py : SVM model code file

>>> Scripts folder : Contains Docker script files




