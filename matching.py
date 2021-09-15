import numpy as np

def matching(etas1, phis1, etas2, phis2):
    '''Match jets with etas2, phis2 (recojets) to jets with etas1, phis1 (myjets), s.t. distance in eta, phi plane (d = sqrt((eta1-eta2)**2 + (phi1-phi2)**2)) is small.
    An array is returned where the element at position i gives the id of the jet2 (recojet) matched to the i-th jet1 (myjet). Priority is given to matching hard jets first.
    When the element is -1, then the considered jet1 (myjet) is assumend to be fake (not a real jet, hence no corresponding jet2 (genjet) exists). 
    When the minimal available distance is larger than 0.4, jet1 is assumed to be fake. Justification for this cut off is given in "DeltaR_cutoff" notebook.'''
    R_cutoff = 0.4
    my_matching = np.full(len(etas1), -1, dtype='int8') # default value of -1 means that all unmatched myjets will be regarded as fake jets
    
    for i, (eta1, phi1) in enumerate(zip(etas1, phis1)):
        # find distance between one of jet1 (myjets) and all jet2 (genjets)
        Rs = np.sqrt((eta1-etas2)**2 + (phi1-phis2)**2)
        
        mins_idx = np.argsort(Rs)
        for idx in mins_idx:
            # when index already used in an earlier pair, go the index providing next smallest distance
            if idx in my_matching:
                continue
            # when difference larger than max difference for correct matching, claim jet to be fake and leave entry as -1 
            elif Rs[idx] >= R_cutoff:
                break
            else:
                my_matching[i] = idx
                break
                
    return my_matching