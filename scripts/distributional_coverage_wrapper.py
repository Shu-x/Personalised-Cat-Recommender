from cornac.metrics import RankingMetric
from recommenders.evaluation.python_evaluation import distributional_coverage
import pandas as pd


class DistributionalCoverage(RankingMetric):
    def __init__(self, k=-1, *args):
        RankingMetric.__init__(self, name="DistributionalCoverage", k=k)

    def compute(self, gt_pos, pd_rank, seen_items, reco_items, **kwargs):
        if len(seen_items) > 0:
            train_df = pd.concat(
                [
                    pd.Series(1, index=range(len(seen_items)), dtype="int64"),
                    pd.Series(seen_items),
                ],
                axis=1,
                keys=["userID", "catID"],
            )
        else:
            train_df = pd.DataFrame(columns=["userID", "catID"])

        rank = [r for r in pd_rank if r not in seen_items]
        reco_df = pd.concat(
            [pd.Series(1, index=range(len(rank)), dtype="int64"), pd.Series(rank)],
            axis=1,
            keys=["userID", "catID"],
        )
        # reco_df["relevance"] = reco_df["catID"].map(
        #     lambda id: reco_items.get(id) if id in reco_items else 0
        # )

        # convert all columns to string
        train_df = train_df.astype(int)
        reco_df = reco_df.astype(int)
        # print(train_df)
        # print(reco_df)
        dist_cov = distributional_coverage(
            train_df,
            reco_df,
            col_user="userID",
            col_item="catID"
        )
        # if len(dist_cov) > 0:
        #     return dist_cov.loc[0, "distributional_coverage"]
        # else:
        #     return 0
        return dist_cov


# def distributional_coverage(
#     train_df, reco_df, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL
# ):
#     """Calculate distributional coverage for recommendations across all users.
#     The metric definition is based on formula (21) in the following reference:

#     :Citation:

#         G. Shani and A. Gunawardana, Evaluating Recommendation Systems,
#         Recommender Systems Handbook pp. 257-297, 2010.

#     Args:
#         train_df (pandas.DataFrame): Data set with historical data for users and items they
#                 have interacted with; contains col_user, col_item. Assumed to not contain any duplicate rows.
#                 Interaction here follows the *item choice model* from Castells et al.
#         reco_df (pandas.DataFrame): Recommender's prediction output, containing col_user, col_item,
#                 col_relevance (optional). Assumed to not contain any duplicate user-item pairs.
#         col_user (str): User id column name.
#         col_item (str): Item id column name.

#     Returns:
#         float: distributional coverage
#     """
#     # In reco_df, how  many times each col_item is being recommended
#     df_itemcnt_reco = pd.DataFrame(
#         {"count": reco_df.groupby([col_item]).size()}
#     ).reset_index()

#     # the number of total recommendations
#     count_row_reco = reco_df.shape[0]

#     df_entropy = df_itemcnt_reco
#     df_entropy["p(i)"] = df_entropy["count"] / count_row_reco
#     df_entropy["entropy(i)"] = df_entropy["p(i)"] * np.log2(df_entropy["p(i)"])

#     d_coverage = -df_entropy.agg({"entropy(i)": "sum"})[0]

#     return d_coverage
        
# def user_serendipity(
#     train_df,
#     reco_df,
#     item_feature_df=None,
#     item_sim_measure=DEFAULT_ITEM_SIM_MEASURE,
#     col_item_features=DEFAULT_ITEM_FEATURES_COL,
#     col_user=DEFAULT_USER_COL,
#     col_item=DEFAULT_ITEM_COL,
#     col_sim=DEFAULT_SIMILARITY_COL,
#     col_relevance=None,
# ):
#     """Calculate average serendipity for each user's recommendations.

#     Args:
#         train_df (pandas.DataFrame): Data set with historical data for users and items they
#               have interacted with; contains col_user, col_item. Assumed to not contain any duplicate rows.
#         reco_df (pandas.DataFrame): Recommender's prediction output, containing col_user, col_item,
#               col_relevance (optional). Assumed to not contain any duplicate user-item pairs.
#         item_feature_df (pandas.DataFrame): (Optional) It is required only when item_sim_measure='item_feature_vector'.
#             It contains two columns: col_item and features (a feature vector).
#         item_sim_measure (str): (Optional) This column indicates which item similarity measure to be used.
#             Available measures include item_cooccurrence_count (default choice) and item_feature_vector.
#         col_item_features (str): item feature column name.
#         col_user (str): User id column name.
#         col_item (str): Item id column name.
#         col_sim (str): This column indicates the column name for item similarity.
#         col_relevance (str): This column indicates whether the recommended item is actually
#               relevant to the user or not.
#     Returns:
#         pandas.DataFrame: A dataframe with following columns: col_user, user_serendipity.
#     """
#     df_user_item_serendipity = user_item_serendipity(
#         train_df,
#         reco_df,
#         item_feature_df,
#         item_sim_measure,
#         col_item_features,
#         col_user,
#         col_item,
#         col_sim,
#         col_relevance,
#     )
#     df_user_serendipity = (
#         df_user_item_serendipity.groupby(col_user)
#         .agg({"user_item_serendipity": "mean"})
#         .reset_index()
#     )
#     df_user_serendipity.columns = [col_user, "user_serendipity"]
#     df_user_serendipity = df_user_serendipity.sort_values(col_user).reset_index(
#         drop=True
#     )

#     return df_user_serendipity