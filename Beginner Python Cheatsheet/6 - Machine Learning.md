
# 6 - Machine Learning

__Table of Contents__
 * Machine Learning
    - [General Format](#general)
    - [Linear Regression](#linear)
    - [Logistic Regression](#log)

<a id="general"></a>
### General Format

<a id="linear"></a>
### Linear Regression

```python
print('Coefficients: \n', lm.coef_) #prints the coefficients of the linear model

coef = pd.DataFrame(lm.coef_,x.columns) #print coefficients of the model in a dataframe
coef.columns = ['Coef']

```

<a id="log"></a>
### Logistic Regression