# A-B-Testing-of-Mobile-Game
The data is taken from a mobile game called Cookie Cat. It represnts players progress throughout the game in which they encounter obstacles represented as gates that players have to cross either by waiting or doing in-app purchases. In this project we conduct an A/B test to study players retention rate if the first gate in the game is moved from level 30 to level 40. 

## Load packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
## Load dataset

```python
data=pd.read_csv('cookie_cats.csv')
data.head()
```

## Inspect data

```python
# Missing values in dataset
data.isnull().sum()

#Number of Unique Values
data.nunique()

# Overall Summary 
data.sum_gamerounds.describe()

# Summary based on Versions
data.groupby('version').sum_gamerounds.agg(['count','median','min','max','mean','std'])

```

## Removing abnormal values

```python
# Making boxplot of both versions.
sns.boxplot(x='version',y='sum_gamerounds',data=data)
plt.title('Boxplot of gate_30 and gate_40 versions')

# Removing abnormal values
data2=data[data.sum_gamerounds < data.sum_gamerounds.max()]

# Making boxplot again after removing outliers
sns.boxplot(x='version',y='sum_gamerounds',data=data2)
plt.title('Boxplot of gate_30 and gate_40 versions after removing outliers')

# Removing abnormal values
data2=data[data.sum_gamerounds < data.sum_gamerounds.max()]

# Number of users who reached gate 30 and 40
data2.groupby('sum_gamerounds').userid.count().loc[[30,40]]
```
## Analyze Retention Details

Retention variables gives us player retention details.

1) retention_1 - did the player come back and play 1 day after installing?
2) retention_7 - did the player come back and play 7 days after installing?

```python
pd.DataFrame({"1Day_count": data2["retention_1"].value_counts(),
              "7Day_count": data2["retention_7"].value_counts(),
              "1Day_ratio": data2["retention_1"].value_counts() / len(data2),
              "7Day_ratio": data2["retention_7"].value_counts() / len(data2)})

# Summarize based on version and retention_1
data2.groupby(['version','retention_1']).sum_gamerounds.\
agg(['count','median','min','max','mean','std'])

# Summarize based on version and retention_7
data2.groupby(['version','retention_7']).sum_gamerounds.\
agg(['count','median','min','max','mean','std'])

# Better suitable position of gate based on sample
data2.groupby('version').retention_1.mean(), data2.groupby('version').retention_7.mean()
```

## A/B Testing

```python
# Define function for A/B Test
def AB_Test(dataframe, group, target):
    
    # Packages
    from scipy.stats import shapiro
    import scipy.stats as stats
    
    # Split A/B
    groupA = dataframe[dataframe[group] == "gate_30"][target]
    groupB = dataframe[dataframe[group] == "gate_40"][target]
    
    # Assumption: Normality
    ntA = shapiro(groupA)[1] < 0.05
    ntB = shapiro(groupB)[1] < 0.05
    # H0: Distribution is Normal! - False
    # H1: Distribution is not Normal! - True
    
    if (ntA == False) & (ntB == False): # "H0: Normal Distribution"
        # Parametric Test
        # Assumption: Homogeneity of variances
        leveneTest = stats.levene(groupA, groupB)[1] < 0.05
        # H0: Homogeneity: False
        # H1: Heterogeneous: True
        
        if leveneTest == False:
            # Homogeneity
            ttest = stats.ttest_ind(groupA, groupB, equal_var=True)[1]
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
        else:
            # Heterogeneous
            ttest = stats.ttest_ind(groupA, groupB, equal_var=False)[1]
            # H0: M1 == M2 - False
            # H1: M1 != M2 - True
    else:
        # Non-Parametric Test
        ttest = stats.mannwhitneyu(groupA, groupB)[1] 
        # H0: M1 == M2 - False
        # H1: M1 != M2 - True
        
    # Result from test
    temp = pd.DataFrame({
        "AB Hypothesis":[ttest < 0.05], 
        "p-value":[ttest]
    })
    temp["Test Type"] = np.where((ntA == False) & (ntB == False), "Parametric", "Non-Parametric")
    temp["AB Hypothesis"] = np.where(temp["AB Hypothesis"] == False, "Fail to Reject H0", "Reject H0")
    temp["Comment"] = np.where(temp["AB Hypothesis"] == "Fail to Reject H0", "gate_30/gate_40 groups are similar!", "gate_30/gate_40 groups are not similar!")
    
    # Columns
    if (ntA == False) & (ntB == False):
        temp["Homogeneity"] = np.where(leveneTest == False, "Yes", "No")
        temp = temp[["Test Type", "Homogeneity","AB Hypothesis", "p-value", "Comment"]]
    else:
        temp = temp[["Test Type","AB Hypothesis", "p-value", "Comment"]]
    
    # Print Hypothesis
    print("# A/B Testing Hypothesis")
    print("H0: A == B")
    print("H1: A != B", "\n")
    return temp
```
    
```python
    # Apply A/B Testing
AB_Test(dataframe=data2, group = "version", target = "sum_gamerounds")
```

#### The Shapiro test conducted above rejected our Null Hypothesis Ho for the normality assumption.Hence we had to conduct a Non-paramteric test such as Mann Whitney U test. By conducting this test we reject our Null Hypothesis Ho. This proves that there is a statistical difference between both the groups which consist of moving the gate from level 30 to level 40.

##### ( For detailed results of the project please refer to the Notebook present in the folder) 
