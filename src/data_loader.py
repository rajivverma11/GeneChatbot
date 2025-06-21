import pandas as pd
import math

def load_product_descriptions(file_path: str, limit: int = 100):
    df = pd.read_csv(file_path, index_col=0)
    product_description = []

    for _, row in df.iterrows():
        if len(product_description) >= limit:
            break
        product = ""
        title = row.get("TITLE")
        if pd.notna(title):
            product += f"Title\n{title}\n"
        description = row.get("DESCRIPTION")
        if pd.notna(description):
            product += f"Description\n{description}\n"
        if product.strip():
            product_description.append(product.strip())

    return product_description
