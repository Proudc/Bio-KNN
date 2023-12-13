from tools import feature_distance

# all_feature distance
def all_feature_distance(all_feature_distance_type,
                         anchor_embedding,
                         positive_embedding,
                         negative_embedding,
                         channel,
                         distance_net = None):
    if all_feature_distance_type == "cosine":
        # cosine_distance
        positive_learning_distance = feature_distance.cosine_torch(anchor_embedding, positive_embedding)
        negative_learning_distance = feature_distance.cosine_torch(anchor_embedding, negative_embedding)
        cross_learning_distance    = feature_distance.cosine_torch(positive_embedding, negative_embedding)
    elif all_feature_distance_type == "euclidean":
        # euclidean_distance
        positive_learning_distance = feature_distance.euclidean_torch(anchor_embedding, positive_embedding)
        negative_learning_distance = feature_distance.euclidean_torch(anchor_embedding, negative_embedding)
        cross_learning_distance    = feature_distance.euclidean_torch(positive_embedding, negative_embedding)
    elif all_feature_distance_type == "manhattan":
        # manhattan_distance
        positive_learning_distance = feature_distance.manhattan_torch(anchor_embedding, positive_embedding)
        negative_learning_distance = feature_distance.manhattan_torch(anchor_embedding, negative_embedding)
        cross_learning_distance    = feature_distance.manhattan_torch(positive_embedding, negative_embedding)
    elif all_feature_distance_type == "mlp":
        # mlp_distance
        positive_learning_distance = distance_net(anchor_embedding, positive_embedding)
        negative_learning_distance = distance_net(anchor_embedding, negative_embedding)
        cross_learning_distance    = distance_net(positive_embedding, negative_embedding)    
    elif all_feature_distance_type == "hyperbolic":
        # hyperbolic_distance
        positive_learning_distance = feature_distance.hyperbolic_torch(anchor_embedding, positive_embedding)
        negative_learning_distance = feature_distance.hyperbolic_torch(anchor_embedding, negative_embedding)
        cross_learning_distance    = feature_distance.hyperbolic_torch(positive_embedding, negative_embedding)
    else:
        raise ValueError('Unsupported All Feature Distance Type: {}'.format(all_feature_distance_type))
    return positive_learning_distance, negative_learning_distance, cross_learning_distance

# cnn_feature distance
def cnn_feature_distance(cnn_feature_distance_type,
                         anchor_cnn_embedding,
                         positive_cnn_embedding,
                         negative_cnn_embedding,
                         channel):
    if cnn_feature_distance_type == "euclidean_sum":
        positive_aln_learning = feature_distance.euclidean_torch_separate_sum(anchor_cnn_embedding, positive_cnn_embedding, channel = channel)
        negative_aln_learning = feature_distance.euclidean_torch_separate_sum(anchor_cnn_embedding, negative_cnn_embedding, channel = channel)
        cross_aln_learning    = feature_distance.euclidean_torch_separate_sum(positive_cnn_embedding, negative_cnn_embedding, channel = channel)
    elif cnn_feature_distance_type == "euclidean_sep":
        positive_aln_learning = feature_distance.euclidean_torch_separate(anchor_cnn_embedding, positive_cnn_embedding, channel = channel)
        negative_aln_learning = feature_distance.euclidean_torch_separate(anchor_cnn_embedding, negative_cnn_embedding, channel = channel)
        cross_aln_learning    = feature_distance.euclidean_torch_separate(positive_cnn_embedding, negative_cnn_embedding, channel = channel)
    elif cnn_feature_distance_type == "manhattan_sum":
        positive_aln_learning = feature_distance.manhattan_torch_separate_sum(anchor_cnn_embedding, positive_cnn_embedding, channel = channel)
        negative_aln_learning = feature_distance.manhattan_torch_separate_sum(anchor_cnn_embedding, negative_cnn_embedding, channel = channel)                
        cross_aln_learning    = feature_distance.manhattan_torch_separate_sum(positive_cnn_embedding, negative_cnn_embedding, channel = channel)
    elif cnn_feature_distance_type == "manhattan_sep":                
        positive_aln_learning = feature_distance.manhattan_torch_separate(anchor_cnn_embedding, positive_cnn_embedding, channel = channel)
        negative_aln_learning = feature_distance.manhattan_torch_separate(anchor_cnn_embedding, negative_cnn_embedding, channel = channel)
        cross_aln_learning    = feature_distance.manhattan_torch_separate(positive_cnn_embedding, negative_cnn_embedding, channel = channel)
    elif cnn_feature_distance_type == "hyperbolic_sum":
        positive_aln_learning = feature_distance.hyperbolic_torch_separate_sum(anchor_cnn_embedding, positive_cnn_embedding, channel = channel)
        negative_aln_learning = feature_distance.hyperbolic_torch_separate_sum(anchor_cnn_embedding, negative_cnn_embedding, channel = channel)                
        cross_aln_learning    = feature_distance.hyperbolic_torch_separate_sum(positive_cnn_embedding, negative_cnn_embedding, channel = channel)
    elif cnn_feature_distance_type == "hyperbolic_sep":                
        positive_aln_learning = feature_distance.hyperbolic_torch_separate(anchor_cnn_embedding, positive_cnn_embedding, channel = channel)
        negative_aln_learning = feature_distance.hyperbolic_torch_separate(anchor_cnn_embedding, negative_cnn_embedding, channel = channel)
        cross_aln_learning    = feature_distance.hyperbolic_torch_separate(positive_cnn_embedding, negative_cnn_embedding, channel = channel)
    else:
        raise ValueError('Unsupported CNN Feature Distance Type: {}'.format(cnn_feature_distance_type))
    return positive_aln_learning, negative_aln_learning, cross_aln_learning