class DataFrame:
    def __init__(self, data: list, columns: list = None):
        """
        Initialize the DataFrame with data and optional column names.
        :param data: List of dictionaries representing rows of data.
        :param columns: List of column names.
        """
        self.data = data
        if columns:
            self.columns = columns
        else:
            # Infer column names from the keys of the first row if columns are not provided
            self.columns = list(data[0].keys()) if data else []

    def __getitem__(self, column_name: str):
        """
        Access a column by its name.
        :param column_name: The column name to access.
        :return: List of values from the specified column.
        """
        if column_name not in self.columns:
            raise KeyError(f"Column '{column_name}' does not exist in DataFrame.")
        return [row[column_name] for row in self.data]

    def append(self, row: dict):
        """
        Append a new row to the DataFrame.
        :param row: Dictionary representing the new row.
        """
        if not all(col in row for col in self.columns):
            raise ValueError(f"Row keys must match the DataFrame columns: {self.columns}")
        self.data.append(row)

    def iterrows(self):
        """
        Iterate over the rows of the DataFrame.
        :return: Iterator yielding index and row dictionaries.
        """
        for idx, row in enumerate(self.data):
            yield idx, row

    def set_column(self, column_name: str, values: list):
        """
        Set or update a column with a list of values.
        :param column_name: The column name to set.
        :param values: List of values to assign to the column.
        """
        if len(values) != len(self.data):
            raise ValueError("Length of values must match the number of rows in DataFrame.")
        for i, row in enumerate(self.data):
            row[column_name] = values[i]
        if column_name not in self.columns:
            self.columns.append(column_name)

    def to_list(self):
        """
        Convert the DataFrame back to a list of dictionaries (rows).
        :return: List of dictionaries representing rows.
        """
        return self.data

    def apply(self, column_name: str, func):
        """
        Apply a function to each element in the specified column.
        :param column_name: The column to apply the function on.
        :param func: The function to apply to each element.
        :return: List of the results of the function applied to each element in the column.
        """
        if column_name not in self.columns:
            raise KeyError(f"Column '{column_name}' does not exist in DataFrame.")

        # Apply the function to each element in the specified column
        results = [func(value) for value in self[column_name]]
        return results

    def sort_values(self, by: str, ascending: bool = True):
        """
        Sort the DataFrame by a specific column.
        :param by: The column name to sort by.
        :param ascending: Whether to sort in ascending order (default: True).
        :return: A new DataFrame sorted by the specified column.
        """
        if by not in self.columns:
            raise KeyError(f"Column '{by}' does not exist in DataFrame.")

        # Sort the data based on the specified column
        sorted_data = sorted(self.data, key=lambda row: row[by], reverse=not ascending)
        return DataFrame(sorted_data, columns=self.columns)

    def head(self, n: int):
        """
        Return the first n rows of the DataFrame.
        :param n: The number of rows to return.
        :return: A new DataFrame with the first n rows.
        """
        return DataFrame(self.data[:n], columns=self.columns)

    def filter(self, column_name: str, value):
        """
        Filter the DataFrame based on a column name and a specific value.
        :param column_name: The column to filter by.
        :param value: The value to match in the column.
        :return: A new DataFrame with rows where column_name == value.
        """
        if column_name not in self.columns:
            raise KeyError(f"Column '{column_name}' does not exist in DataFrame.")

        filtered_data = [row for row in self.data if row[column_name] == value]
        return DataFrame(filtered_data, columns=self.columns)