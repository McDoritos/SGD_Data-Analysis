import pandas as pd
def read_csv(file_name):
    df = pd.read_csv(file_name)
    print(f"File {file_name} read with success!")
    return df

def read_xlsx(file_name, spreadsheet):
    if spreadsheet is None:
        df = pd.read_excel(file_name)
    else:
        df = pd.read_excel(file_name, sheet_name=spreadsheet)
            
    print("Arquivo XLSX lido com sucesso!")
    return df

