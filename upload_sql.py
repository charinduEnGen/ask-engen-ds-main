import re
import pandas as pd

from pathlib import Path
from sqlalchemy import create_engine, URL

# =====================================
# CONFIGURATION
# =====================================

FOLDER_PATH = "/Users/denuwanliya/Documents/EngenuityAI/Orient/OneDrive_1_26-10-2026"

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3307,
    "username": "root",
    "password": "root200202",
    "database": "engenuity_test"
}

# =====================================
# MYSQL CONNECTION
# =====================================

connection_url = URL.create(
    drivername="mysql+pymysql",
    username=DB_CONFIG["username"],
    password=DB_CONFIG["password"],
    host=DB_CONFIG["host"],
    port=DB_CONFIG["port"],
    database=DB_CONFIG["database"]
)

engine = create_engine(connection_url)

# =====================================
# NAME FORMATTING
# =====================================

def clean_name(name):
    name = str(name).strip().lower()
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")

    if not name:
        name = "unnamed"

    if name[0].isdigit():
        name = "table_" + name

    return name[:60]


def clean_columns(df):
    columns = []
    seen = {}

    for col in df.columns:
        name = clean_name(col)

        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0

        columns.append(name)

    df.columns = columns
    return df


# =====================================
# IMPORT DATAFRAME TO MYSQL
# =====================================

def import_to_mysql(df, table_name):

    if df.empty:
        print(f"Skipping empty table: {table_name}")
        return

    df = clean_columns(df)

    # Preserve values as strings, including IDs
    df = df.astype(object).where(pd.notna(df), None)

    df.to_sql(
        name=clean_name(table_name),
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=1000,
        method="multi"
    )

    print(
        f"Imported: {table_name} | "
        f"Rows: {len(df)} | "
        f"Columns: {len(df.columns)}"
    )


# =====================================
# PROCESS ALL FILES
# =====================================

def import_folder(folder_path):

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    supported_extensions = {".csv", ".xlsx", ".xls"}

    files = [
        file for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in supported_extensions
    ]

    print(f"Found {len(files)} files.")

    for file in files:

        print(f"\nProcessing: {file.name}")

        try:

            if file.suffix.lower() == ".csv":

                df = pd.read_csv(
                    file,
                    dtype=str,
                    encoding="utf-8-sig"
                )

                table_name = clean_name(file.stem)

                import_to_mysql(df, table_name)

            elif file.suffix.lower() in [".xlsx", ".xls"]:

                sheets = pd.read_excel(
                    file,
                    sheet_name=None,
                    dtype=str
                )

                for sheet_name, df in sheets.items():

                    table_name = (
                        f"{file.stem}_{sheet_name}"
                    )

                    import_to_mysql(df, table_name)

        except Exception as e:

            print(f"ERROR processing {file.name}: {e}")

    print("\nFolder import completed!")


# =====================================
# EXECUTION
# =====================================

if __name__ == "__main__":

    import_folder(FOLDER_PATH)