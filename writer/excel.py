import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name


def write_to_excel(file_path: str):
    """
    Write the provided data to an Excel file with multiple sheets.
    Each DataFrame in the dictionary will be written to a separate sheet.
    """

    def write_to_excel(data: dict[str, pd.DataFrame]):

        workbook = xlsxwriter.Workbook(
            file_path,
            {
                "nan_inf_to_errors": True,
                "strings_to_numbers": False,
                "strings_to_formulas": True,
                "strings_to_urls": True,
            },
        )

        for key, df in data.items():
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Value for key '{key}' is not a DataFrame.")

            # Create an empty worksheet with the name of the DataFrame
            worksheet = workbook.add_worksheet(key)

            # Custom formatting
            for col_index, col in enumerate(df.columns):
                print(f"Processing column: {col} in sheet: {key}")
                if key == "Print statement_BKT":
                    print("SDF")
                if (
                    df[col]
                    .apply(
                        lambda x: isinstance(x, pd.DataFrame)
                        and len(set(x.columns)) == 2
                        and pd.api.types.is_datetime64_any_dtype(x.iloc[:, 0])
                        and x.iloc[:, 1].dtype == bool
                    )
                    .all()
                ):
                    # Format for boolean time series
                    all_x = pd.concat(
                        [ts_df.iloc[:, 0] for ts_df in df[col]], ignore_index=True
                    )
                    min_x = all_x.min()
                    max_x = all_x.max()
                    x_name = f"x_{worksheet.name}_{col_index}"
                    y_name = f"y_{worksheet.name}_{col_index}"
                    worksheet_x = workbook.add_worksheet(x_name)
                    worksheet_x.hide()
                    worksheet_y = workbook.add_worksheet(y_name)
                    worksheet_y.hide()
                    for row, ts_df in enumerate(df[col]):
                        for c, (dt, val) in enumerate(
                            zip(ts_df.iloc[:, 0], ts_df.iloc[:, 1])
                        ):
                            worksheet_x.write(
                                row + 1,
                                c,
                                (dt - min_x)
                                / (max_x - min_x + pd.Timedelta(seconds=1e-9))
                                * 100.000,
                            )
                            worksheet_y.write(row + 1, c, 1 if val else -1)
                        end_column = xl_col_to_name(len(ts_df.iloc[:, 0]))
                        worksheet.add_sparkline(
                            row + 1,
                            col_index,
                            {
                                "range": f"'{worksheet_y.name}'!$A${row + 2}:${end_column}${row + 2}",
                                "date_axis": f"'{worksheet_x.name}'!$A${row + 2}:${end_column}${row + 2}",
                                "type": "column",
                                "negative_points": True,
                            },
                        )
                    worksheet.set_column(col_index, col_index, 20)
                    df[col] = df[col].apply(
                        lambda x: (
                            "[" + ", ".join(str(i) for i in x) + "]"
                            if isinstance(x, list)
                            else str(x)
                        )
                    )
                    # Write data in cell
                    # worksheet.write_column(
                    #     1,
                    #     col_index,
                    #     df[col]
                    #     .where(pd.notnull(df[col]), "")
                    #     .apply(lambda x: " " * 100 + f"\n{x}"),
                    # )
                elif (
                    df[col]
                    .apply(
                        lambda x: isinstance(x, pd.DataFrame)
                        and len(set(x.columns)) == 2
                        and pd.api.types.is_datetime64_any_dtype(x.iloc[:, 0])
                        and x.iloc[:, 1].dtype == float
                    )
                    .all()
                ):
                    # Format for float time series
                    all_x = pd.concat(
                        [ts_df.iloc[:, 0] for ts_df in df[col]], ignore_index=True
                    )
                    min_x = all_x.min()
                    max_x = all_x.max()
                    x_name = f"x_{worksheet.name}_{col_index}"
                    y_name = f"y_{worksheet.name}_{col_index}"
                    worksheet_x = workbook.add_worksheet(x_name)
                    worksheet_x.hide()
                    worksheet_y = workbook.add_worksheet(y_name)
                    worksheet_y.hide()
                    for row, ts_df in enumerate(df[col]):
                        for c, (dt, val) in enumerate(
                            zip(ts_df.iloc[:, 0], ts_df.iloc[:, 1])
                        ):
                            worksheet_x.write(
                                row + 1,
                                c,
                                (dt - min_x)
                                / (max_x - min_x + pd.Timedelta(seconds=1e-9))
                                * 100.000,
                            )
                            worksheet_y.write(row + 1, c, val)
                        end_column = xl_col_to_name(len(ts_df.iloc[:, 0]))
                        worksheet.add_sparkline(
                            row + 1,
                            col_index,
                            {
                                "range": f"'{worksheet_y.name}'!$A${row + 2}:${end_column}${row + 2}",
                                "date_axis": f"'{worksheet_x.name}'!$A${row + 2}:${end_column}${row + 2}",
                                "type": "line",
                            },
                        )
                    worksheet.set_column(col_index, col_index, 20)
                    df[col] = df[col].apply(
                        lambda x: (
                            "[" + ", ".join(str(i) for i in x) + "]"
                            if isinstance(x, list)
                            else str(x)
                        )
                    )
                    # Write data in cell
                    # worksheet.write_column(
                    #     1,
                    #     col_index,
                    #     df[col]
                    #     .where(pd.notnull(df[col]), "")
                    #     .apply(lambda x: " " * 100 + f"\n{x}"),
                    # )
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    # Format for datetime columns
                    date_format = workbook.add_format(
                        {"num_format": "yyyy-mm-dd hh:mm:ss"}
                    )
                    worksheet.write_column(1, col_index, df[col], date_format)
                    worksheet.set_column(col_index, col_index, 19)
                elif df[col].dtype == "object":
                    # Format for object columns (strings)
                    df[col] = df[col].apply(
                        lambda x: (
                            "[" + ", ".join(str(i) for i in x) + "]"
                            if isinstance(x, list)
                            else str(x)
                        )
                    )
                    worksheet.write_column(
                        1, col_index, df[col].where(pd.notnull(df[col]), "")
                    )
                else:
                    # Default format for other types (e.g., numeric)
                    worksheet.write_column(
                        1, col_index, df[col].where(pd.notnull(df[col]), "")
                    )

            # Format sheet as table
            (max_row, max_col) = df.shape
            if max_row > 0 and max_col > 0:
                last_col = xl_col_to_name(max_col - 1)
                worksheet.add_table(
                    0,
                    0,
                    max_row,
                    max_col - 1,
                    {
                        "columns": [{"header": str(col)} for col in df.columns],
                        "name": f"{key}_table",
                    },
                )

        # Close the workbook
        while True:
            try:
                workbook.close()
                break
            except Exception as e:
                print(f"Error closing workbook: {e}")
                input("Press Enter to retry...")
        print(f"Data successfully written to {file_path}")

    return write_to_excel
