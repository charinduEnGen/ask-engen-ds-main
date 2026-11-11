import re
from pathlib import Path

import pandas as pd
from sqlalchemy import URL, create_engine, text

# =====================================
# CONFIGURATION
# =====================================

FOLDER_PATH = "/Users/denuwanliya/Documents/EngenuityAI/Orient/OneDrive_1_26-10-2026"

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 1521,
    "username": "system",
    "password": "oracle_test_123",
    "service_name": "FREEPDB1",
}

# =====================================
# ORACLE CONNECTION
# =====================================

connection_url = URL.create(
    drivername="oracle+oracledb",
    username=DB_CONFIG["username"],
    password=DB_CONFIG["password"],
    host=DB_CONFIG["host"],
    port=DB_CONFIG["port"],
    query={"service_name": DB_CONFIG["service_name"]},
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

    return name[:30]


def clean_columns(df):
    columns = []
    seen = set()

    for col in df.columns:
        base = clean_name(col)
        name = base
        counter = 1

        while name.upper() in seen:
            suffix = f"_{counter}"
            name = f"{base[:30-len(suffix)]}{suffix}"
            counter += 1

        seen.add(name.upper())
        columns.append(name)

    df.columns = columns
    return df


# =====================================
# IMPORT DATAFRAME TO ORACLE
# =====================================

def import_to_oracle(df, table_name):

    if df.empty:
        print(f"Skipping empty table: {table_name}")
        return

    df = clean_columns(df)

    # Preserve values as strings, including IDs
    df = df.astype(object).where(pd.notna(df), None)

    safe_table_name = clean_name(table_name)

    df.to_sql(
        name=safe_table_name,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=500,
        dtype=None,
    )

    print(
        f"Imported: {safe_table_name} | "
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
        and not file.name.startswith("~$")
    ]

    print(f"Found {len(files)} files.")

    for file in files:

        print(f"\nProcessing: {file.name}")

        try:

            if file.suffix.lower() == ".csv":

                df = pd.read_csv(
                    file,
                    dtype=str,
                    encoding="utf-8-sig",
                )

                import_to_oracle(df, file.stem)

            elif file.suffix.lower() in {".xlsx", ".xls"}:

                sheets = pd.read_excel(
                    file,
                    sheet_name=None,
                    dtype=str,
                )

                for sheet_name, df in sheets.items():

                    table_name = f"{file.stem}_{sheet_name}"

                    import_to_oracle(df, table_name)

        except Exception as e:

            print(f"ERROR processing {file.name}: {e}")

    print("\nFolder import completed!")


# =====================================
# EXECUTION
# =====================================

if __name__ == "__main__":

    try:

        with engine.connect() as connection:
            connection.execute(text("SELECT 1 FROM dual"))

        print("Oracle connection successful.")

        import_folder(FOLDER_PATH)

    except Exception as e:

        print(f"Oracle connection or import failed: {e}")

    finally:
        engine.dispose()