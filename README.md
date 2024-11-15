# Predicting Heart Disease Risk with Logistic Regression Techniques

## Project Overview
This project implements logistic regression techniques to predict the risk of heart disease based on various health metrics and patient data. The model aims to assist healthcare professionals in early diagnosis and intervention.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Work Flow](#work-flow)
- [Dataset](#dataset)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation
To set up the project, clone the repository and install the required packages using the following commands:

```bash
git clone https://github.com/shafaq-aslam/Predicting-Heart-Disease-Risk-with-Logistic-Regression.git
cd Predicting-Heart-Disease-Risk-with-Logistic-Regression
pip install -r requirements.txt
```
## Usage
Run the Jupyter Notebook for training and evaluating the logistic regression model:

```bash
jupyter notebook {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "11554612-8d8c-45e1-b847-07ed34acff74",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Importing Necessary Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "9969754d-28b9-4b3c-a925-08e3daa26478",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import pylab as pl\n",
    "import numpy as np\n",
    "import scipy.optimize as opt\n",
    "import statsmodels.api as sm\n",
    "from sklearn import preprocessing\n",
    "'exec(% matplotlib inline)'\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.mlab as mlab\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "80513706-7be9-4fb1-9205-99783784fe90",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loading the dataset\n",
    "# framingham.csv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "332d1e7f-7687-4588-9012-a43282d7e8c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# dataset\n",
    "disease_df = pd.read_csv(r\"C:\\Users\\HP\\Heart-Disease-Prediction-Using-Logistic-Regression\\framingham.csv\")\n",
    "disease_df.drop(['education'], inplace = True, axis = 1)\n",
    "disease_df.rename(columns ={'male':'Sex_male'}, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "92b8e98d-5f0b-4a27-a70b-960543e88555",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Handling Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "4dab8010-e636-4077-918b-d29287b58537",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Sex_male  age  currentSmoker  cigsPerDay  BPMeds  prevalentStroke  \\\n",
      "0         1   39              0         0.0     0.0                0   \n",
      "1         0   46              0         0.0     0.0                0   \n",
      "2         1   48              1        20.0     0.0                0   \n",
      "3         0   61              1        30.0     0.0                0   \n",
      "4         0   46              1        23.0     0.0                0   \n",
      "\n",
      "   prevalentHyp  diabetes  totChol  sysBP  diaBP    BMI  heartRate  glucose  \\\n",
      "0             0         0    195.0  106.0   70.0  26.97       80.0     77.0   \n",
      "1             0         0    250.0  121.0   81.0  28.73       95.0     76.0   \n",
      "2             0         0    245.0  127.5   80.0  25.34       75.0     70.0   \n",
      "3             1         0    225.0  150.0   95.0  28.58       65.0    103.0   \n",
      "4             0         0    285.0  130.0   84.0  23.10       85.0     85.0   \n",
      "\n",
      "   TenYearCHD  \n",
      "0           0  \n",
      "1           0  \n",
      "2           0  \n",
      "3           1  \n",
      "4           0   (3751, 15)\n",
      "TenYearCHD\n",
      "0    3179\n",
      "1     572\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# removing NaN / NULL values\n",
    "disease_df.dropna(axis = 0, inplace = True)\n",
    "print(disease_df.head(), disease_df.shape)\n",
    "print(disease_df.TenYearCHD.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "b962cf38-98bb-4b29-a0b2-45db73d09cbf",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting the Dataset into Test and Train Sets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "b8240121-675f-4d3d-8040-c0e65091517f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train set: (2625, 6) (2625,)\n",
      "Test set: (1126, 6) (1126,)\n"
     ]
    }
   ],
   "source": [
    "X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', \n",
    "                           'totChol', 'sysBP', 'glucose']])\n",
    "y = np.asarray(disease_df['TenYearCHD'])\n",
    "\n",
    "# normalization of the dataset\n",
    "X = preprocessing.StandardScaler().fit(X).transform(X)\n",
    "\n",
    "# Train-and-Test -Split\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split( \n",
    "        X, y, test_size = 0.3, random_state = 4)\n",
    "\n",
    "print ('Train set:', X_train.shape,  y_train.shape)\n",
    "print ('Test set:', X_test.shape,  y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "697607c8-9279-4eda-8bb6-99aefe9f1d2d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Exploratory Data Analysis of Heart Disease Dataset\n",
    "# Ten Year’s CHD Record of all the patients available in the dataset:\n",
    "# Counting number of patients affected by CHD where (0= Not Affected; 1= Affected)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "cc743ddf-1b2e-4d41-8c7b-ea5e5c957ad5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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DTtnzc6dO5WcnKzs7Gx77LPPPrvo3AMHDlS/fv30+OOPa8KECY3um6usrGy2++beeecdHThwoMVfs0KYAwAAl23hwoVKT09XbGys7rrrLnXo0EEffPCBDhw4oEcfffSK5g4NDdXs2bM1Y8YMNTQ06Oabb1ZVVZUKCwt1zTXXKDMzUzfccIPWr1+vt956S3FxcdqwYYP27dt30atgDodDa9euVUpKim655RbNnz9fffr0UXV1tf76179q27ZtKigouOye6+rqVFZWpvr6ep04cUJ5eXnKzc1Venq6fvnLXzb1o7gkhDkAAHDZRo8erb/97W9avHixli5dqqCgIPXp06fZntr8/e9/r8jISOXm5upf//qXrr32WvuVH5J03333qbi4WBMmTJDD4dDEiROVnZ2trVu3XnTun/zkJ9q/f78ee+wxTZkyRV9++aWio6OVnJysFStWNKnfvLw8RUdHKzAwUF26dNGAAQP05JNPKjMz85xPzTYnh3X2TkBcUFVVlVwulzwej8LCwlr8eDzNCrQsnmYFvnX69GmVlJQoLi7O5143tLwLffaXkzt4aTAAAIDBCHMAAAAGI8wBAAAYjDAHAABgMMIcAACAwQhzAAAABiPMAQAAGIwwBwAAYDDCHAAAgMH8+nNeq1at0qpVq/T5559Lkm688UYtXLhQaWlpkiTLsrRo0SI988wzqqio0ODBg/Xf//3fuvHGG+056urqNHv2bP3lL39RbW2tbr31Vj399NPq3r27XVNRUaHp06fr9ddflySNHTtWTz31VLP9kC4AADi/1v5Vo6b+wsvTTz+tZcuWqbS0VDfeeKNWrFihoUOHNm9zLcCvV+a6d++u//zP/9T+/fu1f/9+jRw5Uj/72c/04YcfSpKWLl2qJ554QitXrtS+ffvkdrs1atQonTp1yp4jJydHmzdv1qZNm7Rr1y5VV1crPT1d9fX1ds2kSZNUXFysvLw85eXlqbi4WBkZGa1+vgAAoG166aWXlJOTowULFuj999/X0KFDlZaWpqNHj/q7tYtqc7/NGh4ermXLlunXv/61YmJilJOTo7lz50r69ipcVFSU/vCHP+jee++Vx+NRt27dtGHDBk2YMEGS9MUXXyg2NlZvvvmmRo8erUOHDqlfv37avXu3Bg8eLEnavXu3kpKS9M9//lO9e/e+pL74bVbg6sJvswLfao3f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",
      "text/plain": [
       "<Figure size 700x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# counting no. of patients affected with CHD\n",
    "plt.figure(figsize=(7, 5))\n",
    "sns.countplot(x='TenYearCHD', data=disease_df,\n",
    "             palette=\"muted\", hue='TenYearCHD')\n",
    "plt.show()\n",
    "\n",
    "# This code is modified by Susobhan Akhuli"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "8432c217-7a16-4072-8501-9aa7646e20d9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "colors = sns.color_palette(\"Set2\")\n",
    "laste = disease_df['TenYearCHD'].plot(color=colors)\n",
    "plt.show(laste)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "fe78199d-3fd8-42c3-8ab9-b172fb1b4130",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fitting Logistic Regression Model for Heart Disease Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "01de3061-761b-423f-961e-441b6362e390",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "logreg = LogisticRegression()\n",
    "logreg.fit(X_train, y_train)\n",
    "y_pred = logreg.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "d8cc3b5a-edf0-47c0-9865-90f53af15978",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluating Logistic Regression Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "248bdb49-aa5f-4d8d-87ed-8a4bf8399469",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy of the model is = 0.8490230905861457\n"
     ]
    }
   ],
   "source": [
    "# Evaluation and accuracy\n",
    "from sklearn.metrics import accuracy_score\n",
    "print('Accuracy of the model is =', \n",
    "      accuracy_score(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "5bcf3cc9-800a-4e51-94d7-77516cdeedfe",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Confusion Matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "edbf8dcf-d64a-4bef-86ff-be366954a424",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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e7R8+XJt3bpVH330kZYvX6577rlHjz76qObNm1eesQIAAOAHfG6tTp48WTNnztTAgQNL3OvZs6datWql8ePH6ze/+U1AAwQAACgNu1b9SOSOHz+u++6776r327ZtqxMnTgQkKAAAgGvhQGA/Wqt333235s6de9X78+bN09133x2QoAAAAHBtPlfkpkyZom7dumnt2rXq0qWL4uPjZbPZ5HQ6tX79eh05ckRr1qwpz1gBAAA8aK36kci1a9dO+/bt06xZs7R9+3Y5nU5JUkJCgrp3764nn3xSderUKa84AQAAvJi82zRQ/DpHrk6dOpo4cWJ5xQIAAAA/8BVdAADASPYQvpCgTDXJAQMGaPTo0V5jf/jDHzRgwICABAUAAHAtdltIwC5Tlakil52dLbfb7TV2/PhxHTt2LCBBAQAA4NrKlMht3ry5xNiiRYt+cjAAAAC+orXKM3IAAMBQ7Fr1MZF77733fF4wJSWlzMEAAADAdz4lcj179vRpMZvNpuLi4p8SDwAAgE9orfqYyF25sQEAAADBxzNyAADASCYfGxIoZUrk8vPzlZmZqaNHj6qwsNDr3tChQwMSGAAAwI+htVqGRG737t3q2rWrLl68qPz8fMXExOj06dOKjIxUXFwciRwAAEAF8bsm+cwzz+ihhx7SmTNnFBERoe3bt+vIkSNq0aKFXn755fKIEQaoElFZ054ap6/f3q6L73+lj195Vy3valLq3NnDJshan6NhvxjoGasWdYtee/pF/WN+pvL/9586suRTvTr4BVWNjKqoPwKAnyCiSoSenvYbvfP1fK29uFLTP56sei2Tgx0WbnD2kJCAXabyuyKXlZWlOXPmyG63y263y+VyqW7dupo0aZL69++vXr16lUecuM69MXyyGtWpp8cnDtOJb7/RYx17acOkd9Rw4P068a3TM69H2wfUukEzHT/t9Hp9Ymy8EmPjNXLui/riyD9VO/5WzR42QYmx8Xr4xUEV/ccB4Kdn3xiipEa1lfH4FJ0+cUadH+uglze8pF83HKzTJ74Ndni4QdFaLUNFLiwsTDabTZIUHx+vo0ePSpKio6M9/42bS6XwSur9X101at54fbT3Ux068bX+tHiqsp3H9NRDj3vmJcYmaMbvXlK/jCEqulTktcb+r7/UL1/4rd7fvkGHTx7R5qxtGr1goh76z058UIHrXHilcP2s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",
      "text/plain": [
       "<Figure size 800x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The details for confusion matrix is =\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.85      0.99      0.92       951\n",
      "           1       0.61      0.08      0.14       175\n",
      "\n",
      "    accuracy                           0.85      1126\n",
      "   macro avg       0.73      0.54      0.53      1126\n",
      "weighted avg       0.82      0.85      0.80      1126\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "conf_matrix = pd.DataFrame(data = cm, \n",
    "                           columns = ['Predicted:0', 'Predicted:1'], \n",
    "                           index =['Actual:0', 'Actual:1'])\n",
    "\n",
    "plt.figure(figsize = (8, 5))\n",
    "sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = \"PRGn\")\n",
    "\n",
    "plt.show()\n",
    "print('The details for confusion matrix is =')\n",
    "print (classification_report(y_test, y_pred))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```
