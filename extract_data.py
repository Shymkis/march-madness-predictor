from bs4 import BeautifulSoup
import pandas as pd

for year in range(2002, 2026):
    with open(f"html_tables/{year}.txt", "r") as file:
        content = file.read()
    soup = BeautifulSoup(content, "html.parser")
    table = soup.find("table")

    # Extract data from the table
    data = []
    for row in table.find_all("tr"):
        if not row.find(class_="next_left").find(class_="seed"):
            continue  # Skip rows without a seed
        row_data = []
        for cell in row.find_all("td"):
            row_data.append(cell.get_text(strip=True))
        data.append(row_data)

    # Convert data to a DataFrame
    df = pd.DataFrame(data)
    df.columns = [
        "NetRtg.Rk",
        "Team", "Conf", "W-L",
        "NetRtg",
        "ORtg", "ORtg.Rk",
        "DRtg", "DRtg.Rk",
        "AdjT", "AdjT.Rk",
        "Luck", "Luck.Rk",
        "SOS.NetRtg", "SOS.NetRtg.Rk",
        "SOS.ORtg", "SOS.ORtg.Rk",
        "SOS.DRtg", "SOS.DRtg.Rk",
        "NCSOS.NetRtg", "NCSOS.NetRk"
    ]

    for col in ["NetRtg", "ORtg", "DRtg", "AdjT", "Luck", "SOS.NetRtg", "SOS.ORtg", "SOS.DRtg", "NCSOS.NetRtg"]:
        df[col] = df[col].replace({'^\+': '', '^-': '-'}, regex=True).astype(float)

    df['Team'] = df['Team'].str.replace('*', '', regex=False)
    df['Seed'] = df['Team'].str.extract(r'(\d+)$')
    df['Team'] = df['Team'].str.replace(r'\d+$', '', regex=True).str.strip()
    df[['W', 'L']] = df['W-L'].str.split('-', expand=True)
    df = df.drop(columns=['W-L'])
    for col in ["W", "L", "Seed"]:
        df[col] = df[col].astype(int)
    df["WinPct"] = df["W"] / (df["W"] + df["L"])

    df.to_csv(f"csv_files/{year}.csv", index=False)