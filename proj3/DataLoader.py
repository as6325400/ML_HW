class DataLoader():

    def __init__(self, path: str) -> None:
        """
        The meaning of varibles:
        data -> list, n * 2 matrix, n = nums of data points
        """
        self.data = self.read_data(path)

    def read_data(self, path: str) -> list:
        """
        To read data coordinate points and put them into 'list' data type
        """
        with open(path, 'r') as raw_data:
            lines = raw_data.readlines()
            data = []       # dataset

            for words in lines:
                words = words.strip().split(' ')
                for i in range(2):
                    words[i] = float(words[i])  # type: ignore

                data.append(words[0:2])

            return data  # type: ignore
