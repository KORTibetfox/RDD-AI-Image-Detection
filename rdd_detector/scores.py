"""Pure score arithmetic shared by the detector and lightweight tests."""


def image_score(original, blurred, rotated, blurred_rotated, rotation_weight=0.5):
    return original - blurred - rotation_weight * rotated + rotation_weight * blurred_rotated


def combine_scores(image, latent):
    return image * latent ** 2


def classify(score, threshold):
    """Experimental decision convention from the source notebook, not a probability."""
    return "real" if score > threshold else "ai_generated"

