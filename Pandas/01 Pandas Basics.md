# Pandas Basics

Author: methylDragon  
Contains a syntax reference and code snippets for Pandas!  
It's a collection of code snippets and tutorials from everywhere all mashed together!       

------

## Pre-Requisites

### Required

- Python knowledge, this isn't a tutorial!
- Pandas installed
  
  - I'll assume you've already run these lines as well 
  
    ```python
    import numpy as np
    import pandas as pd
    ```



## Table Of Contents <a name="top"></a>

1. [Introduction](#1)    
2. [Pandas Basics](#2)    
   2.1 [Data Types](#2.1)    
   2.2 [Series Basics](#2.2)    
   2.3 [DataFrame Basics](#2.3)    
   2.4 [Panel Basics](#2.4)    
   2.5 [Catagorical Data](#2.5)    
   2.6 [Basic Binary Operations](#2.6)    
   2.7 [Casting and Conversion](#2.7)    
   2.8 [Conditional Indexing](#2.8)    
   2.9 [IO](#2.9)    
   2.10 [Plotting](#2.10)    
   2.11 [Sparse Data](#2.11)    
3. [Series Operations](#3)    
   3.1 [Manipulating Series Text](#3.1)    
   3.2 [Time Series](#3.2)    
   3.3 [Time Deltas](#3.3)    
4. [DataFrame Operations](#4)    
   4.1 [Preface](#4.1)    
   4.2 [Iterating Through DataFrames](#4.2)    
   4.3 [Sorting, Reindexing, and Renaming DataFrame Values](#4.3)    
   4.4 [Replacing DataFrame Values](#4.4)    
   4.5 [Function Application on DataFrames](#4.5)    
   4.6 [Descriptive Statistics](#4.6)    
   4.7 [Statistical Methods](#4.7)    
   4.8 [Window Functions](#4.8)    
   4.9 [Data Aggregation](#4.9)    
   4.10 [Dealing with Missing Data](#4.10)    
   4.11 [GroupBy Operations](#4.11)    
   4.12 [Merging and Joining](#4.12)    
   4.13 [Concatenation](#4.13)    
5. [EXTRA: Helpful Notes](#5)    




## 1. Introduction <a name="1"></a>

> *pandas* is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the [Python](https://www.python.org/) programming language.
>
> *pandas* is a [NumFOCUS](https://www.numfocus.org/open-source-projects.html) sponsored project. This will help ensure the success of development of *pandas* as a world-class open-source project, and makes it possible to [donate](https://pandas.pydata.org/donate.html) to the project.
> 
> <https://pandas.pydata.org/>

This document will list the most commonly used functions in Pandas, to serve as a reference when using it.

It's not meant to be exhaustive, merely acting as a quick reference for the syntax for basic operations with Pandas. Please do not hesitate to consult the [official documentation](<https://pandas.pydata.org/pandas-docs/stable) for pandas for more in-depth dives into the library!

**Key Features**

Source: <https://www.tutorialspoint.com/python_pandas/python_pandas_introduction.htm>

- Fast and efficient DataFrame object with default and customized indexing.
- Tools for loading data into in-memory data objects from different file formats.
- Data alignment and integrated handling of missing data.
- Reshaping and pivoting of date sets.
- Label-based slicing, indexing and subsetting of large data sets.
- Columns from a data structure can be deleted or inserted.
- Group by data for aggregation and transformations.
- High performance merging and joining of data.
- Time Series functionality.

---

Install it!

```shell
# Best to use conda
$ conda install pandas

# But it's possible to use the PyPI wheels as well
$ pip install pandas
```

You might also need to install additional dependencies

```shell
$ sudo apt-get install python-numpy python-scipy python-matplotlibipythonipythonnotebook
python-pandas python-sympy python-nose
```



If you need additional help or need a refresher on the parameters, feel free to use:

```python
help(pd.FUNCTION_YOU_NEED_HELP_WITH)
```

---

**Credits:**

A lot of these notes I'm adapting from 

<https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html>

<https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python>

<https://www.tutorialspoint.com/python_pandas/python_pandas_introduction.htm>



## 2. Pandas Basics <a name="2"></a>

### 2.1 Data Types <a name="2.1"></a>
[go to top](#top)


Note that Pandas is built on top of Numpy.

There are three types of data structures that Pandas deals with:

- Series
  - 1D labelled homogeneous array, size-immutable
  - If heterogeneous data is entered, the data-type will become 'object'
- DataFrame
  - Contains series data
  - 2D labelled, size-mutable, table structure
  - Potentially heterogeneous columns
- Panel
  - Contains DataFrames
  - 3D labelled, size-mutable array

**The major focus of this syntax reference will deal with DataFrames**. Since they're the most commonly manipulated objects when Pandas is concerned.



### 2.2 Series Basics <a name="2.2"></a>
[go to top](#top)


> A Series is a one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc.). The axis labels are collectively called index.
>
> <https://www.tutorialspoint.com/python_pandas/python_pandas_series.htm>

![Image result for pandas series](assets/series-and-dataframe.width-1200.png)

[Image Source](<https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/>)

#### **Creating Series Objects**

```python
# Empty Series
s = pd.Series()

# Series from ndarray
s = pd.Series(np.array([1, 2, 3]))
s = pd.Series(np.array([1, 2, 3]), index=[100, 101, 102]) # With custom indexing

# Series from Dict
# Dictionary keys are used to construct the index
s = pd.Series({'a': 0, 'b': 1, 'c': 2})

# Series from scalar
s = pd.Series(5, index=[0, 1, 2]) # Creates 3 rows of value 5
```

#### **Accessing Values**

```python
# By position
s[0]

# By index
s['index_name']

# By slice
s[-3:] # Retrieves last 3 elements

# Fancy indexing works also!
s[[0, 1, 2]]
s[['index_1', 'index_2', 'index_3']]

# Head and Tail
s.head()
s.tail()
s.head(5) # First 5
s.tail(5) # Last 5
```

#### **Series Properties**

```python
s.axes 		# Returns list of row axis labels
s.dtype 	# Returns data type of entries
s.empty 	# True if series is empty
s.ndim 		# Dimension. 1 for series
s.size 		# Number of elements
s.values 	# Returns the Series as an ndarray
```



### 2.3 DataFrame Basics <a name="2.3"></a>
[go to top](#top)


> A Data frame is a two-dimensional data structure, i.e., data is aligned in a tabular fashion in rows and columns.
>
> <https://www.tutorialspoint.com/python_pandas/python_pandas_dataframe.htm>

![Structure Table](assets/structure_table.jpg)

Image Source: <https://www.tutorialspoint.com/python_pandas/python_pandas_dataframe.htm>

#### **Creating DataFrame Objects**

```python
# Empty DataFrame
df = pd.DataFrame()

# DataFrame from List
df = pd.DataFrame([1, 2, 3, 4, 5]) # Single Column
df = pd.DataFrame([['a', 1], ['b', 2]], columns=['name_1', 'name_2']) # Multi columns
df = pd.DataFrame([1, 2, 3], dtype=float) # Convert the ints to floats

# DataFrame from Series
df = s.to_frame()

# DataFrame from Dict of Lists
df = pd.DataFrame({'Name':['methylDragon', 'toothless', 'smaug'], 'Rating': [10, 5, 2]})

# DataFrame from List of Dicts
df = pd.DataFrame([{'Name': 'methylDragon', 'Rating': 10},
                   {'Name': 'toothless', 'Rating': 5},
                   {'Name': 'smaug'}]) # NaN will be appended for missing values

# DataFrame from Dict of Series
# Similarly, NaN will be added for missing values
df = pd.DataFrame({'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
                   'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])})

# Creating with Non-Default Index
df = pd.DataFrame([1, 2, 3], index=['a', 'b', 'c'])
```

#### **Important Note on Mutability**

**NOTE:** Most operations will **not** change the original DataFrame unless the DataFrame is **reassigned**, or you use an `inplace=True` flag, which changes the DataFrame in question in place.

#### **Basic Operations**

**Column**

```python
# Column Selection
df['column_name']
df.column_name # This also works! (Only if the column name is a string though..)

# Column Selection by dtype
df.select_dtypes(include=[dtypes])

# Adding a new Column
df['new_column_name'] = pd.Series([1, 2, 3])

# Deleting a Column (Either one works)
del df['column_name']
df.pop(['column_name'])

# Math for Columns
df['column_1'] + df['column_2'] # Gives you a new column that is the addition of the first two
```

**Row**

```python
# Row Selection by Label
df.loc['row_lable/index']

# Row Selection by Position Index
df.iloc[0] # Selects first row

# Row Slicing
df[-3:]

# Adding Rows
df.append(df2)
df.append(df2, ignore_index=True) # To ignore indices

# Deleting Rows
df.drop('label_to_drop')

# Deleting rows with None/NaN/empty values
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
df.dropna(axis=0, how='any') # Drop rows with any column containing None
df.dropna(axis=0, how='all') # Drop rows with all columns containing None
df.dropna(axis=0, thresh=2) # Drop rows with 2 or more columns containing None

# Head and Tail
df.head()
df.tail()
df.head(5) # First 5 rows
df.tail(5) # Last 5 rows
```

#### **DataFrame Properties**

```python
df.T 		# Transpose
df.axes 	# Row axis and column axis labels
df.dtypes 	# Data types of elements
df.empty 	# True if empty
df.ndim 	# Dimension (number of axes)
df.shape 	# Tuple representing the shape (dimensionality) of the DataFrame
df.size 	# Number of elements
df.values 	# Numpy represendation, NDFrame
```



### 2.4 Panel Basics <a name="2.4"></a>
[go to top](#top)


> A **panel** is a 3D container of data. The term **Panel data** is derived from econometrics and is partially responsible for the name pandas − **pan(el)-da(ta)**-s.
>
> The names for the 3 axes are intended to give some semantic meaning to describing operations involving panel data. They are −
>
> - **items** − axis 0, each item corresponds to a DataFrame contained inside.
> - **major_axis** − axis 1, it is the index (rows) of each of the DataFrames.
> - **minor_axis** − axis 2, it is the columns of each of the DataFrames.
>
> <https://www.tutorialspoint.com/python_pandas/python_pandas_panel.htm>

#### **Creating Panel Objects**

```python
# Empty Panel
p = pd.Panel()

# Panel from 3D ndarray
p = pd.Panel(np.random.rand(2, 4, 5))

# Panel from dict of DataFrames
data = {'Item1' : pd.DataFrame(np.random.randn(4, 3)), 
        'Item2' : pd.DataFrame(np.random.randn(4, 2))}
p = pd.Panel(data)
```

#### **Accessing Values**

```python
# By Item
p['Item1'] # Gives you the corresponding dataframe

# By Major Axis
p.major_xs(1) # Shows all data from the second row across all dataframes

'''
Eg: If the panel's first item is as such:
            0          1          2
0    0.488224  -0.128637   0.930817
>> 1    0.417497   0.896681   0.576657 <<
2   -2.775266   0.571668   0.290082
3   -0.400538  -0.144234   1.110535

Then the Output of p.major_xs(1) is:
      Item1
0   0.417497
1   0.896681
2   0.576657

It's a transpose of the second row's elements (of the original DataFrame)!
'''

# By Minor Axis
p.minor_xs(1)

'''
Eg: Same deal as above, same first item

Output of p.minor_xs(1) are the items under the second column (of the original DataFrame)!

       Item1
0   -0.128637
1    0.896681
2    0.571668
3   -0.144234
'''
```



### 2.5 Catagorical Data <a name="2.5"></a>
[go to top](#top)


So imagine you have data that's made of a limited number of actual values

Eg: [1, 1, 1, 3, 2, 3, 2, 1, 2, 3, 1]

There's a way to encode the fact that there are only three kinds of values - Catagories!

#### **Construct Catagorical Data**

```python
# Source: https://www.tutorialspoint.com/python_pandas/python_pandas_categorical_data.htm

s = pd.Series(["a","b","c","a"], dtype="category")
'''
Output

0  a
1  b
2  c
3  a
dtype: category
Categories (3, object): [a, b, c]
'''

# Generate just a list-like object with catagories
cat = pd.Categorical(['a', 'b', 'c', 'a', 'b', 'c'])
# [a, b, c, a, b, c]
# Categories (3, object): [a, b, c]

# Or do it with stated catagories!
cat = pd.Categorical(['a','b','c','a','b','c','d'], ['c', 'b', 'a'])
# [a, b, c, a, b, c, NaN]
# Categories (3, object): [c, b, a]

# Specify catagories with ordered catagories
# This one implies c < b < a
cat = pd.Categorical(['a','b','c','a','b','c','d'], ['c', 'b', 'a'],ordered=True)
```

#### **Properties and Altering Catagories**

```python
df.describe() # For general
s.categories() # Find catagories
s.ordered() #
s.cat.categories() # Use this to edit the categories

# Add catagories
s = s.cat.add_categories([4])

# Remove catagories
s.cat.remove_categories("a")

# Compare catagories
# You may compare catagorical data, aligned by category
cat = pd.Series([1,2,3]).astype("category", categories=[1,2,3], ordered=True)
cat1 = pd.Series([2,2,2]).astype("category", categories=[1,2,3], ordered=True)

cat > cat1
'''
Output

0  False
1  False
2  True
dtype: bool
'''
```



### 2.6 Basic Binary Operations <a name="2.6"></a>
[go to top](#top)


https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.add.html#pandas.DataFrame.add

#### **Arithmetic**

```python
df.add(other)
df.sub(other)
df.mul(other)
df.div(other)
df.truediv(other) # For floats
df.floordiv(other) # For integers
df.mod(other)
df.pow(other)
df.divmod(other) # Returns tuple of (quotient, remainder)

df.radd(other) # Reverse
df.rsub(other) # Reverse

# You may specify fill-values for missing values too!
df.add(other, fill_value=0)
```

#### **Boolean Reductions**

```python
(df > 0).all()

#  empty, any, all, bool all work.

# You can also do comparisons! (Eg. ==, >, etc.)
```



### 2.7 Casting and Conversion <a name="2.7"></a>
[go to top](#top)


```python
# Casting object to dtype
df.astype(dtype)
df.astype(dtype, copy=False) # Do not return a copy

# Attempt to infer better dtype for object columns
df.convert_objects(convert_dates=True) # Unconvertibles become NaT
df.convert_objects(convert_numeric=True) # Unconvertibles become NaN
```



### 2.8 Conditional Indexing <a name="2.8"></a>
[go to top](#top)


So you remember that fancy indexing works?

```python
# Now you can do it with conditions too!
df[df > 0]
df.where(df > 0)
```



### 2.9 IO <a name="2.9"></a>
[go to top](#top)


<https://pandas.pydata.org/pandas-docs/version/0.20/io.html>

| Format Type | Data Description                                             | Reader                                                       | Writer                                                       |
| :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| text        | [CSV](https://en.wikipedia.org/wiki/Comma-separated_values)  | [read_csv](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-read-csv-table) | [to_csv](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-store-in-csv) |
| text        | [JSON](http://www.json.org/)                                 | [read_json](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-json-reader) | [to_json](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-json-writer) |
| text        | [HTML](https://en.wikipedia.org/wiki/HTML)                   | [read_html](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-read-html) | [to_html](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-html) |
| text        | Local clipboard                                              | [read_clipboard](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-clipboard) | [to_clipboard](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-clipboard) |
| binary      | [MS Excel](https://en.wikipedia.org/wiki/Microsoft_Excel)    | [read_excel](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-excel-reader) | [to_excel](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-excel-writer) |
| binary      | [HDF5 Format](https://support.hdfgroup.org/HDF5/whatishdf5.html) | [read_hdf](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-hdf5) | [to_hdf](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-hdf5) |
| binary      | [Feather Format](https://github.com/wesm/feather)            | [read_feather](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-feather) | [to_feather](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-feather) |
| binary      | [Msgpack](http://msgpack.org/index.html)                     | [read_msgpack](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-msgpack) | [to_msgpack](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-msgpack) |
| binary      | [Stata](https://en.wikipedia.org/wiki/Stata)                 | [read_stata](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-stata-reader) | [to_stata](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-stata-writer) |
| binary      | [SAS](https://en.wikipedia.org/wiki/SAS_(software))          | [read_sas](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-sas-reader) |                                                              |
| binary      | [Python Pickle Format](https://docs.python.org/3/library/pickle.html) | [read_pickle](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-pickle) | [to_pickle](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-pickle) |
| SQL         | [SQL](https://en.wikipedia.org/wiki/SQL)                     | [read_sql](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-sql) | [to_sql](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-sql) |
| SQL         | [Google Big Query](https://en.wikipedia.org/wiki/BigQuery)   | [read_gbq](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-bigquery) | [to_gbq](https://pandas.pydata.org/pandas-docs/version/0.20/io.html#io-bigquery) |

```python
# Custom Indexing
pd.read_csv("file", index_col=['index_col_name'])

# With converted datatypes
pd.read_csv("file", dtype={'col': dtype})

# Column names
pd.read_csv("file", names=['1', 'b', 'etc'])

# Skip rows
pd.read_csv("file", skiprows=2)
```



### 2.10 Plotting <a name="2.10"></a>
[go to top](#top)


Source: <https://www.tutorialspoint.com/python_pandas/python_pandas_visualization.htm>

```python
df.plot() 						# Line plot
df.plot.bar() 					# Bar chart
df.plot.bar(stacked=True) 		# Stacaked bar chart
df.plot.barh() 					# Horizontal bar chart
df.plot.barh(stacked=True) 		# Horizontal stacked bar chart
df.plot.hist(bins=20) 			# Plot histogram
df.diff.hist(bins=30) 			# Plot different histograms for each column
df.plot.box() 					# Bot plot
df.plot.area()					# Area plot
df.plot.scatter(x='a', y='b') 	# Scatter plot
df.plot.pie(subplots=True)		# Pit plot
```



### 2.11 Sparse Data <a name="2.11"></a>
[go to top](#top)


You can sparsify data to save on space on Disk or in the interpretor memory!

```python
# Sparsify
sparse_obj = obj.to_sparse() # Default sparsifies NaN/missing
sparse_obj = obj.to_sparse(fill_value=0) # Sparsify target value

# Convert back
sparse_obj.to_dense()

# Properties
sparse_obj.density
```



## 3. Series Operations <a name="3"></a>

### 3.1 Manipulating Series Text <a name="3.1"></a>
[go to top](#top)


Source: <https://www.tutorialspoint.com/python_pandas/python_pandas_working_with_text_data.htm>

| 1    | **lower()**Converts strings in the Series/Index to lower case. |
| ---- | ------------------------------------------------------------ |
| 2    | **upper()**Converts strings in the Series/Index to upper case. |
| 3    | **len()**Computes String length().                           |
| 4    | **strip()**Helps strip whitespace(including newline) from each string in the Series/index from both the sides. |
| 5    | **split(' ')**Splits each string with the given pattern.     |
| 6    | **cat(sep=' ')**Concatenates the series/index elements with given separator. |
| 7    | **get_dummies()**Returns the DataFrame with One-Hot Encoded values. |
| 8    | **contains(pattern)**Returns a Boolean value True for each element if the substring contains in the element, else False. |
| 9    | **replace(a,b)**Replaces the value **a** with the value **b**. |
| 10   | **repeat(value)**Repeats each element with specified number of times. |
| 11   | **count(pattern)**Returns count of appearance of pattern in each element. |
| 12   | **startswith(pattern)**Returns true if the element in the Series/Index starts with the pattern. |
| 13   | **endswith(pattern)**Returns true if the element in the Series/Index ends with the pattern. |
| 14   | **find(pattern)**Returns the first position of the first occurrence of the pattern. |
| 15   | **findall(pattern)**Returns a list of all occurrence of the pattern. |
| 16   | **swapcase**Swaps the case lower/upper.                      |
| 17   | **islower()**Checks whether all characters in each string in the Series/Index in lower case or not. Returns Boolean |
| 18   | **isupper()**Checks whether all characters in each string in the Series/Index in upper case or not. Returns Boolean. |
| 19   | **isnumeric()**Checks whether all characters in each string in the Series/Index are numeric. Returns Boolean. |

#### **Example**

```python
s.str.lower()
```



### 3.2 Time Series <a name="3.2"></a>
[go to top](#top)


```python
# Get Current Time
pd.datetime.now() # Get current time

# Get Time from Timestamp
pd.Timestamp('2019-03-01')
pd.Timestamp(1587687575, unit='s')

# Get a date range
pd.date_range("11:00", "13:30", freq="H").time
pd.date_range("11:00", "13:30", freq="30min").time # Different frequency
# Output:
# [datetime.time(11, 0) datetime.time(11, 30) datetime.time(12, 0)
#  datetime.time(12, 30) datetime.time(13, 0) datetime.time(13, 30)]

# Convert Time Series to Timestamps
pd.to_datetime(SOME_DATETIME_SERIES)
```



### 3.3 Time Deltas <a name="3.3"></a>
[go to top](#top)


These are almost exactly like the datetime library's timedelta objects.

```python
pd.Timedelta(6, unit='h')
pd.Timedelta(days=-2)
pd.Timedelta('2 days 2 hours 15 minutes 30 seconds') # Or even from a string!

# Or from a series
pd.to_timedelta(s)
```



## 4. DataFrame Operations <a name="4"></a>

### 4.1 Preface <a name="4.1"></a>
[go to top](#top)


Even though this section is supposed to be focused on DataFrames, a lot of these operations can be applied to Series and Panel objects as well! It's just that a large part of using Pandas is working with DataFrames

To get at least some brief understanding of your data you can

```python
# Look at the first few rows of data
df.head()

# Look at essential details (like dimensions, data types, etc.)
df.info()
```



### 4.2 Iterating Through DataFrames <a name="4.2"></a>
[go to top](#top)


```python
df.iteritems() # (key, value) pairs (Get by columns)
df.iterrows() # (index, series) pairs (Get by rows)
df.itertuples() # Iterate over rows as named tuples
```



### 4.3 Sorting, Reindexing, and Renaming DataFrame Values <a name="4.3"></a>
[go to top](#top)


```python
# Sort by Values
df.sort_values('column_name', inplace=True) # Sort by values in column

# Sort by Index
df.sort_index(ascending=False) # Default is ascending=True
df.sort_index(axis=1) # Sort by column index

# Reset Index
df.reset_index(inplace=True, drop=True) # Reset index, skip inserting old index as a column

# Rename Columns
df.rename(columns=newcol_names, inplace=True)

# Rename Index
df.rename(index={'index_element_1': 'new_name'})

# Reindex
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.reindex.html
df.reindex(index=[1, 2, 3], columns=[1, 2, 3])

# Reindex to match another dataframe
df.reindex_like(df2)
df.reindex_like(df2, method="ffill") # Fill missing values
# pad/ffill: Forward fill
# bfill/backfill: Backward fill
# nearest: Nearest index value fill
```



### 4.4 Replacing DataFrame Values <a name="4.4"></a>
[go to top](#top)


```python
# Replace strings with numbers
df.replace(['Awful', 'Poor', 'OK', 'Acceptable', 'Perfect'], [0, 1, 2, 3, 4]) 

# Replace using regex
df.replace({'\n': '<br>'}, regex=True)

# Removing Substrings
df['column_name'] = df['column_name'].map(lambda x: x.lstrip('+-').rstrip('aAbBcC'))
```



### 4.5 Function Application on DataFrames <a name="4.5"></a>
[go to top](#top)


```python
# Apply function to all values in a scope
df['column_name'].apply(function_name)

# Apply function to all values in DataFrame
df.applymap(function_name)
```



### 4.6 Descriptive Statistics <a name="4.6"></a>
[go to top](#top)


You can do a bunch of basic statistical calculations on the rows of a DataFrame!

```python
# Sum along axis
# axis=0 : Along columns
# axis=1 : Along rows
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sum.html
df.sum() # Default axis is 0
df.sum(axis=1)
df.sum(axis=0, skipna=True, numeric_only=True, min_count=0)

# Even more!
df.count()		# Number of non-null observations
df.mean()		# Mean of Values
df.median()		# Median of Values
df.mode()		# Mode of values
df.std()		# Standard Deviation of the Values
df.min()		# Minimum Value
df.max()		# Maximum Value
df.abs()		# Absolute Value
df.prod()		# Product of Values
df.cumsum()		# Cumulative Sum
df.cumprod()	# Cumulative Product

# Or just call all of them at once!
df.describe()
```



### 4.7 Statistical Methods <a name="4.7"></a>
[go to top](#top)


```python
# Calculate percentage change
df.pct_change() # Column wise
df.pct_change(axis=1) # Row wise

# Covariance
s.cov(s2) # For series
df.cov() # For frame (calculates covariance between all columns)

# Correlation
df.corr() # For frames
df['col_1'].corr(df['col_2']) # For series

# Data Ranking (Series)
# https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.Series.rank.html
# Check the docs for tie-breaking methods
# average, min, max, first (Default method='average')
s.rank()
```



### 4.8 Window Functions <a name="4.8"></a>
[go to top](#top)


```python
# Rolling Window
df_rolling = df.rolling(window=3)

# Now you can use the window!
# You may use all the descriptive stats and statistical methods
df_rolling.sum()
df_rolling.mean()
df_rolling.median()
df_rolling.std()
# and so on...

# Expanding Window
# (Yields the value of the statistic with all the data available up to that point in time)
df_expanding = df.expanding(min_periods=1)

# Exponential Weighted Functions
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
# You can specify decay, half-life, etc. Check the docs!
df.ewm()
```



### 4.9 Data Aggregation <a name="4.9"></a>
[go to top](#top)


```python
# Basically custom operations on windows!
df_rolling.aggregate(FUNCTION) # On Whole DF
df_rolling['col'].aggregate(FUNCTION) # On Single Column
df_rolling[['col', 'col2']].aggregate(FUNCTION) # On Multiple Columns

# Multiple functions (You'll get two columns as output)
df_rolling.aggregate([FUNCTION_1, FUNCTION_2])

# Multiple functions, on different columns
df_rolling.aggregate({'col_1': FUNCTION_1, 'col_2': FUNCTION_2})

# If you don't run it on a rolling window, it reduces the dimensionality of the data!
df.aggregate(np.sum) # Sums the entire column
```



### 4.10 Dealing with Missing Data <a name="4.10"></a>
[go to top](#top)


Null values can be NA, NaN, NaT, or None.

- NaN: Not a Number
- NaT: Not a Time

```python
# Detect Missing Values
df.isnull() # Gives True if value is null
df.notnull() # Gives True if value is not null

# Filling Missing Data With Scalar
df.fillna(scalar_number)

# Filling Missing Data
# pad/fill: Fills forward
# bfill/backfill: Fills backwards
df.fillna(method='pad')

# Drop Missing Values
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
df.dropna(axis=0, how='any') # Drop rows with any column containing None
df.dropna(axis=0, how='all') # Drop rows with all columns containing None
df.dropna(axis=0, thresh=2) # Drop rows with 2 or more columns containing None
```



### 4.11 GroupBy Operations <a name="4.11"></a>
[go to top](#top)


Source: <https://www.tutorialspoint.com/python_pandas/python_pandas_groupby.htm>

You can group data within your DataFrames in order to:

- Split the DF
- Apply a function to the DF
  - Aggregation
  - Transformation
  - Filtration
- Combine certain results

```python
# Group Data
df_grouped = df.groupby('key') # By column
df_grouped = df.groupby('key', axis=1) # By row
df_grouped = df.groupby(['col_1', 'col_2']) # Multi-Column Group

# View the groups!
df_grouped.groups

# You can iterate through grouped dfs as well!
for i in df_grouped:
    pass

# Select a Single Group
df_grouped.get_group('group_name')

# Apply Aggregations
df_grouped.agg(function)
df_grouped.agg([function_1, function_2])

# Apply Transformations
# Transforms groups or columns inside the dataframe
transformation_function = lambda x: (x - x.mean()) / x.std()*10
df_grouped.transform(transformation_function)

# Apply Filters
# Works like the native Python filter(filtering_function, iterable) !
df_grouped.filter(filtering_function)
df_grouped.filter(lambda x: len(x) > = 3)
```



### 4.12 Merging and Joining <a name="4.12"></a>
[go to top](#top)


> Pandas provides a single function, **merge**, as the entry point for all standard database join operations between DataFrame objects −
>
> `pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=True)`
>
> Here, we have used the following parameters −
>
> - **left** − A DataFrame object.
> - **right** − Another DataFrame object.
> - **on** − Columns (names) to join on. Must be found in both the left and right DataFrame objects.
> - **left_on** − Columns from the left DataFrame to use as keys. Can either be column names or arrays with length equal to the length of the DataFrame.
> - **right_on** − Columns from the right DataFrame to use as keys. Can either be column names or arrays with length equal to the length of the DataFrame.
> - **left_index** − If **True,** use the index (row labels) from the left DataFrame as its join key(s). In case of a DataFrame with a MultiIndex (hierarchical), the number of levels must match the number of join keys from the right DataFrame.
> - **right_index** − Same usage as **left_index** for the right DataFrame.
> - **how** − One of 'left', 'right', 'outer', 'inner'. Defaults to inner. Each method has been described below.
> - **sort** − Sort the result DataFrame by the join keys in lexicographical order. Defaults to True, setting to False will improve the performance substantially in many cases.
>
> <https://www.tutorialspoint.com/python_pandas/python_pandas_merging_joining.htm>

```python
# Code source: https://www.tutorialspoint.com/python_pandas/python_pandas_merging_joining.htm

# Merge two DFs via key
left = pd.DataFrame({
   'id':[1,2,3,4,5],
   'Name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
   'subject_id':['sub1','sub2','sub4','sub6','sub5']})
right = pd.DataFrame({
	'id':[1,2,3,4,5],
   'Name': ['Billy', 'Brian', 'Bran', 'Bryce', 'Betty'],
   'subject_id':['sub2','sub4','sub3','sub6','sub5']})

pd.merge(left,right,on='id')

'''
OUTPUT

Name_x   id  subject_id_x   Name_y   subject_id_y
0  Alex      1          sub1    Billy           sub2
1  Amy       2          sub2    Brian           sub4
2  Allen     3          sub4     Bran           sub3
3  Alice     4          sub6    Bryce           sub6
4  Ayoung    5          sub5    Betty           sub5
'''

# Merge two DFs via multiple keys
pd.merge(left, right, on=['key_1', 'key_2']) # Unmerged values are discarded

# Merge using 'HOW'
'''
Merge Method	SQL Equivalent		Description
left			LEFT OUTER JOIN		Use keys from left object
right			RIGHT OUTER JOIN	Use keys from right object
outer			FULL OUTER JOIN		Use union of keys
inner			INNER JOIN			Use intersection of keys
'''
pd.merge(left, right, on='key', how='left')
```

Join Intuitions

![Image result for outer join inner join image](assets/hMKKt.jpg)

Image source: <https://stackoverflow.com/questions/38549/what-is-the-difference-between-inner-join-and-outer-join>



### 4.13 Concatenation <a name="4.13"></a>
[go to top](#top)


> Pandas provides various facilities for easily combining together **Series, DataFrame**, and **Panel** objects.
>
> ` pd.concat(objs,axis=0,join='outer',join_axes=None, ignore_index=False)`
>
> - **objs** − This is a sequence or mapping of Series, DataFrame, or Panel objects.
> - **axis** − {0, 1, ...}, default 0. This is the axis to concatenate along.
> - **join** − {‘inner’, ‘outer’}, default ‘outer’. How to handle indexes on other axis(es). Outer for union and inner for intersection.
> - **ignore_index** − boolean, default False. If True, do not use the index values on the concatenation axis. The resulting axis will be labeled 0, ..., n - 1.
> - **join_axes** − This is the list of Index objects. Specific indexes to use for the other (n-1) axes instead of performing inner/outer set logic.
>
> <https://www.tutorialspoint.com/python_pandas/python_pandas_concatenation.htm>

```python
# Concatenate DFs
pd.concat([one, two]) # Adds the rows of two DFs together
pd.concat([one, two], keys=['x', 'y']) # This gives keys to each specific DF
pd.concat([one, two], ignore_index=True) # You can also make it ignore the original index

# Concatenate using Append
one.append(two)
```



## 5. EXTRA: Helpful Notes <a name="5"></a>

I couldn't find a suitable place to put this information, so I'll put it here:

- Pivot tables, stacking, and unstacking
  <https://nikgrozev.com/2015/07/01/reshaping-in-pandas-pivot-pivot-table-stack-and-unstack-explained-with-pictures/>
- Package configuration
  <https://www.tutorialspoint.com/python_pandas/python_pandas_options_and_customization.htm>



```
                            .     .
                         .  |\-^-/|  .    
                        /| } O.=.O { |\
```

​    

------

 [![Yeah! Buy the DRAGON a COFFEE!](../assets/COFFEE%20BUTTON%20%E3%83%BE(%C2%B0%E2%88%87%C2%B0%5E).png)](https://www.buymeacoffee.com/methylDragon)

