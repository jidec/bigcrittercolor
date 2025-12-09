def group_values_by_indices(values, indices):
    groups = {}
    for value, index in zip(values, indices):
        if index not in groups:
            groups[index] = []
        groups[index].append(value)

    # Sort the dictionary by its keys and return the values
    return [groups[key] for key in sorted(groups)]


# Example usage
values = [1, 2, 3, 4, 5, 6, 7, 8]
indices = [3, 3, 1, 1, 2, 2, 2, 4]
print(group_values_by_indices(values, indices))