def model_loss(primary_loss, cross_loss, beta):
    """single model: primary loss + β·cross loss."""
    return primary_loss + beta * cross_loss


def branch_loss(model_a_loss, model_b_loss, alignment_loss, gamma):
    """multi-branch model: model_a_loss + model_b_loss + γ·alignment_loss."""
    return model_a_loss + model_b_loss + gamma * alignment_loss