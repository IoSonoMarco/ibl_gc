from lib.models.utils import sigmoid

def contrast_to_sensory_reliability(contrast, slope=2, midpoint=0.125):
    '''
    Naka-Rushton Psychometric function to turn stimulus contrast into sensory evidence.
    
    contrast: detectability of the stimulus in [0,1]
    slope: Naka-Rushton exponential for the psychometric curve
    '''
    c = contrast**slope
    c50 = midpoint**slope
    rel = c/(c + c50 + 1e-16)
    return rel

def generate_sensory_evidence(
    stimulus_side, 
    stimulus_contrast, 
    contrast_slope: float = 2.,
    contrast_midpoint: float = 0.125
):
    '''
    Create a vector in [-1, 1] of signed stimulus sensory evidence (reliability)
    '''
    sensory_reliability = contrast_to_sensory_reliability(stimulus_contrast, contrast_slope, contrast_midpoint)
    return sensory_reliability * (2 * stimulus_side - 1)

def generate_choice_probs_stimulus_contrast(
        mu2_hat, stimulus_side, stimulus_contrast, 
        beta_prior: float = 1., beta_sens: float = 1., 
        contrast_slope: float = 2, contrast_midpoint: float = 0.125,
        lapse: float = 0
):
    '''
    mu2_hat: prior belief about log-odds of stimulus derived from HGF perceptual model
    stimulus_side: 0 (left), or 1 (right)
    stimulus_contrast: detectability of the stimulus ([0, 0.0625, 0.125, 0.25, 1])
    beta_prior: mu2_hat-driven choice weight
    beta_sens: stimulus contrast-driven choice weight
    slope: Naka-Rushton exponential for the psychometric curve of stimulus contrast
    lapse: stimulus-independent mistakes weight
    '''
    sensory_evidence = generate_sensory_evidence(stimulus_side, stimulus_contrast, contrast_slope, contrast_midpoint)
    theta = beta_prior*mu2_hat + beta_sens*sensory_evidence
    choice_probs = (1 - lapse) * sigmoid(theta) + lapse*0.5
    return choice_probs