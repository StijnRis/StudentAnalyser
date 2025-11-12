import os

import pandas as pd


def write_to_csv(folder_path: str):
    """
    Write the provided data to multiple csv file in a folder.
    Each DataFrame in the dictionary will be written to a separate file.
    """

    def write_to_csv(data: dict[str, pd.DataFrame]):
        # Create folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Write each DataFrame to a separate CSV file
        for sheet_name, df in data.items():
            file_path = f"{folder_path}/{sheet_name}.csv"
            df.to_csv(
                file_path,
                index=False,
                encoding="utf-8-sig",
            )

    return write_to_csv
