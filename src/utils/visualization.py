import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_histogram(df, column, hue=None, title=None):
    """Plot a histogram of a numerical column in a DataFrame, for each value of a categorical variable.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to be plotted.
        column (str): The name of the column to be plotted.
        hue (str): The name of the categorical column to be used for grouping the data.
        title (str): The title to be displayed on the plot.

    Returns:
        None

    Example:
        >>> plot_histogram(df, "Age", "Survived", "Distribution of Age by Survival")
    """
    grid = sns.FacetGrid(df, col=hue, hue=hue, height=5)
    grid.map(sns.histplot, column, kde=False, bins=20)


def plot_correlation_matrix(corr, title):
    """Plot correlation matrix in the data.

    Args:
        corr (pandas.DataFrame): Dataframe containing the correlation matrix.
        title (str): Title of the plot.
    """
    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, annot=True, ax=ax)
    ax.set_title(title)
    plt.show()

def plot_count_plot(df, column, hue, title):
    """Plot count plot in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        column (str): Column to plot.
        hue (str): Column to split the plot.
        title (str): Title of the plot.
    """
    # Plot count plot
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.countplot(x=column, hue=hue, data=df, ax=ax)
    ax.set_title(title)
    plt.show()

def plot_cat_plot(df, x_column, y_column, hue=None, col=None, title=None):
    """Plot cat plot in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        x_column (str): Column to plot on x-axis.
        y_column (str): Column to plot on y-axis.
        hue (str): Column to split the plot.
        title (str): Title of the plot.
    """
    # Plot cat plot
    sns.catplot(x=x_column, y=y_column, hue=hue, col=col, data=df, kind='point', estimator=np.mean)
    plt.show()

def plot_violin_plot(df, x_column, y_column, hue, title):
    """Plot violin plot in the data.
    
    Args:
        df (pandas.DataFrame): Dataframe containing the data.
        x_column (str): Column to plot on x-axis.
        y_column (str): Column to plot on y-axis.
        hue (str): Column to split the plot.
        title (str): Title of the plot.
    """
    # Plot violin plot
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(x=x_column, y=y_column, hue=hue, data=df, split=True, ax=ax)
    ax.set_title(title)
    plt.show()