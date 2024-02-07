def rank_scores(scores):
    """Rank the scores within a set, lower scores get higher ranks."""
    sorted_scores = sorted(scores, reverse=True)
    ranks = [sorted_scores.index(score) + 1 for score in scores]
    return ranks


def aggregate_ranks(ranks_set1, ranks_set2, ranks_set3):
    """Aggregate ranks for each value and sort by total rank."""
    aggregated_ranks = [sum(ranks) for ranks in zip(ranks_set1, ranks_set2, ranks_set3)]
    value_ranks_pairs = list(zip(range(2, 9), aggregated_ranks))
    sorted_by_ranks = sorted(value_ranks_pairs, key=lambda x: x[1])
    return sorted_by_ranks


def rank_values_by_scores(scores_set1, scores_set2, scores_set3):
    # Rank the scores within each set
    ranks_set1 = rank_scores(scores_set1)
    ranks_set2 = rank_scores(scores_set2)
    ranks_set3 = rank_scores(scores_set3)

    # Aggregate and sort the ranks
    sorted_values_ranks = aggregate_ranks(ranks_set1, ranks_set2, ranks_set3)

    return sorted_values_ranks


# Example usage:
scores_set1 = [10, 20, 30, 40, 50, 60, 70]
scores_set2 = [5, 15, 25, 35, 45, 55, 65]
scores_set3 = [20, 30, 40, 50, 60, 70, 80]

# Rank the values
ranked_values = rank_values_by_scores(scores_set1, scores_set2, scores_set3)
print(ranked_values[0][0])

# Print ranked values and their total ranks
for value, total_rank in ranked_values:
    print(f"Value: {value}, Total Rank: {total_rank}")