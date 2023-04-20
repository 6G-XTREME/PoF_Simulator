import matplotlib.pyplot as plt
import seaborn as sns

def sciplot(x, y, xlabel='', ylabel='', title='', xlim=None, ylim=None, xscale='linear', yscale='linear', grid=True, style='whitegrid', legend=True, **kwargs):
    """
    Function to generate a scientific plot with Matplotlib and Seaborn, with tweaks for improved output in scientific papers.

    Parameters:
    x (array-like): x-coordinates of the data points
    y (array-like): y-coordinates of the data points (can be a single array or a list of arrays for multiple traces)
    xlabel (str): label for the x-axis (default: '')
    ylabel (str): label for the y-axis (default: '')
    title (str): title of the plot (default: '')
    xlim (tuple): limits for the x-axis (default: None)
    ylim (tuple): limits for the y-axis (default: None)
    xscale (str): scale of the x-axis ('linear' or 'log', default: 'linear')
    yscale (str): scale of the y-axis ('linear' or 'log', default: 'linear')
    grid (bool): whether to show grid lines (default: True)
    style (str): style of the plot ('whitegrid', 'darkgrid', 'white', 'dark', or a custom Seaborn style, default: 'whitegrid')
    legend (bool): whether to show a legend (default: True)
    **kwargs: additional keyword arguments to pass to the Matplotlib plot function, such as 'label' for a legend entry
    """
    # Set the font size and style
    plt.rc('font', size=12)
    plt.rc('font', family='sans-serif')
    plt.rc('font', serif='Arial')
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    # Set the Seaborn style
    sns.set(style=style)

    # Create the plot
    plt.figure(figsize=(6, 4))
    if isinstance(y, list):
        # If y is a list of arrays, plot each array as a separate trace
        for i, y_i in enumerate(y):
            plt.plot(x, y_i, label=kwargs.get('label')[i])
    else:
        # If y is a single array, plot it as a single trace
        plt.plot(x, y, **kwargs)

    # Set the x-axis label and limits
    plt.xlabel(xlabel)
    if xlim is not None:
        plt.xlim(xlim)
    plt.xscale(xscale)

    # Set the y-axis label and limits
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.yscale(yscale)

    # Set the title and grid lines
    plt.title(title)
    plt.grid(grid)

    # Add legend if specified
    if legend:
        plt.legend()

    # Save the plot as a PNG file
    #plt.tight_layout()
    #plt.savefig('plot.png', dpi=300)

    # Show the plot
    plt.show()
