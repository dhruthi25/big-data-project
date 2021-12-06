import matplotlib.pyplot as plt
# line 1 points
x = [i for i in range(1,14)]
accuracy=[23.2,
20.8,
70.0,
68.8,
78.0,
15.6,24.8,25.6,26.0,15.6,22.8,21.6,71.6]
precision=[i*100 for i in [0.23952095808383234,
0.28415300546448086,
0.6167664670658682,
0.6117021276595744,
0.6820809248554913,
0.23353293413173654,
0.29245283018867924,
0.3132530120481928,
0.32065217391304346,
0.21666666666666667,
0.30158730158730157,
0.2857142857142857,
0.634020618556701
]]
recall=[i*100 for i in [0.38095238095238093,0.06837606837606838,0.0,0.8571428571428571,0.8333333333333334,0.9141630901287554]]
f1score=[i*100 for i in [0.9383886255924171,
0.06477732793522267,
0.0,
0.923076923076923,
0.9090909090909091,
0.9551569506726457]]
# plotting the line 1 points 
plt.plot(x, accuracy, label = "accuracy")
# line 2 points
y2 = []
# plotting the line 2 points 
plt.plot(x, recall, label = "recall")
# line 3 points
y3 = []
# plotting the line 3 points 
plt.plot(x, precision, label = "precision")
#line 4 points
y4 =[]
# plotting the line 4 points 
plt.plot(x, f1score, label = "f1 score")


plt.xlabel('batch_no')
# Set the y axis label of the current axis.
plt.ylabel('metric')
# Set a title of the current axes.
plt.title(' Naive Bayes Multinomial Model Batch Size 500')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()