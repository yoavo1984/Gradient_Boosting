def compute_node_goodness(node, tree_features_dict):
    if node.is_leaf():
        return

    goodness = node.error - (node.right_descendant.error + node.left_descendant.error)
    j = node.j
    if j not in tree_features_dict:
        tree_features_dict[j] = 0
    tree_features_dict[j] += goodness

    compute_node_goodness(node.left_descendant, tree_features_dict)
    compute_node_goodness(node.right_descendant, tree_features_dict)


def get_tree_feature_importance(tree):
    tree_features_dict = {}
    compute_node_goodness(tree.root, tree_features_dict)
    return tree_features_dict


def get_features_importance(tree_ensemble):
    features_dict = {}
    trees_vec = []
    for tree in tree_ensemble:
        tree_feature_importance = get_tree_feature_importance(tree)
        trees_vec.append(tree_feature_importance)
        features_dict = {key: features_dict.get(key, 0) + tree_feature_importance.get(key, 0) for key in set(features_dict) | set(tree_feature_importance)}
    # maybe need to divide each feature by number of trees
    normalize_importance(features_dict)
    return features_dict


def normalize_importance(features_dict):
    max_key = max(features_dict, key=features_dict.get)
    max_val = features_dict[max_key]
    for key in features_dict:
        features_dict[key] /= max_val
