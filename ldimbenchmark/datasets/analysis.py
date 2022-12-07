from ldimbenchmark.datasets import Dataset


class DatasetAnalyzer:
    """
    Analyze a dataset
    """

    def __init__(self, outFolder: str):
        self.outFolder = outFolder

    def compare(self, datasets: list[Dataset]):
        """
        Compare the datasets
        """

    def analyze(self, dataset: Dataset):
        """
        Analyze the dataset
        """

        # original_dataset = pd.read_csv(dataset_source_dir, index_col="Timestamp")

        # plot = original_dataset["J-02"].plot()
        # normalDataset["J-02"].plot(ax=plot.axes)
        # uniformDataset["J-02"].plot(ax=plot.axes)

        # first_figure = plt.figure()
        # first_figure_axis = first_figure.add_subplot()
        # first_figure_axis.plot(noise)

        # first_figure = plt.figure()
        # first_figure_axis = first_figure.add_subplot()
        # count, bins, ignored = first_figure_axis.hist(noise, 30, density=True)
        # sigma = 0.01 / 3
        # mu = 0
        # first_figure_axis.plot(
        #     bins,
        #     1
        #     / (sigma * np.sqrt(2 * np.pi))
        #     * np.exp(-((bins - mu) ** 2) / (2 * sigma**2)),
        #     linewidth=2,
        #     color="r",
        # )
