import pandas as pd

from sklearn.model_selection import train_test_split


def make_dataset():
    order_products = pd.read_csv("../data/order_products.csv")

    # consolidate order_products
    total_quantity = (
        order_products.groupby(["order_id", "buy_unit"]).sum().reset_index()
    )
    total_quantity = total_quantity.pivot(
        index="order_id", columns="buy_unit", values="quantity"
    ).reset_index()
    num_products = (
        order_products.groupby(["order_id"])["product_id"].count().reset_index()
    )
    total_order_products = pd.merge(num_products, total_quantity, on=["order_id"])
    total_order_products.rename(
        columns={"product_id": "num_products", "KG": "kg", "UN": "un"}, inplace=True
    )

    orders = pd.read_csv("../data/orders.csv")
    shoppers = pd.read_csv("../data/shoppers.csv")
    storebranch = pd.read_csv("../data/storebranch.csv")
    df = pd.merge(total_order_products, orders, on=["order_id"])
    df = pd.merge(df, shoppers, on=["shopper_id"])
    df = pd.merge(df, storebranch, on=["store_branch_id"])
    df = df.drop(["shopper_id", "store_branch_id", "store_id"], axis=1)

    # drop rows that do not contain the target
    indices = df[df["total_minutes"].isna()].index
    df = df.drop(indices)

    # train-test split
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv("../data/train.csv", index=False)
    test.to_csv("../data/test.csv", index=False)


if __name__ == "__main__":
    make_dataset()
