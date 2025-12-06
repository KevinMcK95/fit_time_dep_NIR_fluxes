import numpy as np
from tqdm import tqdm

def fit_fluxes_optimal(obs_fluxes,read_errs,
                       n_repeat=2,b_vect=None):
    '''
    takes observed fluxes (shape of n_pixels,n_reads-1) 
    and returns the optimal extracted flux (as defined by Brandt 2024)

    inputs:
        obs_fluxes: observed counts, units of electrons, shape (n_pixels,n_reads)
        read_errs: read noise per pixel, units of electrons, shape (n_pixels)
        n_repeat: number of times to update fit (should be small, like 2)
        b_vect: optional, factor to multiply fluxes by, length n_reads-1. If None, then set to all ones

    outputs:
        f_means: the flux means for each pixel
        f_ivars: the flux ivars for each pixel
        chi2s: chi-square of data-to-model comparison for each pixel
    '''

    n_reads = obs_fluxes.shape[1]
    n_pixels = obs_fluxes.shape[0]
    read_vars = np.power(read_errs,2)
    if type(b_vect) is type(None):
        b_vect = np.ones(n_reads-1)

    #define read covariance matrix 
    #(i.e. 2*read_var on diagonal, -1*read_var on one-from-diagonal)
    read_Vs = np.zeros((n_pixels,n_reads-1,n_reads-1))
    for p_ind in range(n_pixels):
        read_Vs[p_ind,np.arange(n_reads-1),np.arange(n_reads-1)] = 2*read_vars[p_ind]
        for j in range(len(read_Vs[p_ind])-1):
            read_Vs[p_ind,j,j+1] = -read_vars[p_ind]
            read_Vs[p_ind,j+1,j] = -read_vars[p_ind]

    obs_flux_diffs = np.diff(obs_fluxes,axis=1)
    data_V = np.zeros((n_pixels,n_reads-1,n_reads-1))

    #first guess
    curr_f_guess = np.maximum(np.nanmedian(obs_flux_diffs,axis=1),0)

    for r_ind in range(n_repeat):
    
        #calculate the best f given the best b
        curr_data_diffs = np.maximum(curr_f_guess[:,None] * b_vect[None,:],0)
        for p_ind in range(n_pixels):
            data_V[p_ind] = np.diag(curr_data_diffs[p_ind])
        comb_V_inv = np.linalg.inv(data_V + read_Vs)
    
        f_ivars = np.einsum('i,ni->n',b_vect,np.einsum('nij,j->ni',comb_V_inv,b_vect))
        f_means = (1/f_ivars) * np.einsum('i,ni->n',b_vect,np.einsum('nij,nj->ni',comb_V_inv,obs_flux_diffs))

        curr_f_guess = np.maximum(f_means,0)

    #calculate the best f given the best b
    curr_model = f_means[:,None] * b_vect[None,:]
    curr_data_diffs = np.maximum(curr_model,0)
    for p_ind in range(n_pixels):
        data_V[p_ind] = np.diag(curr_data_diffs[p_ind])
    comb_V_inv = np.linalg.inv(data_V + read_Vs)

    diff = obs_flux_diffs-curr_model
    chi2s = np.einsum('ni,ni->n',diff,np.einsum('nij,nj->ni',comb_V_inv,diff))

    return f_means,f_ivars,chi2s



def first_guess_parameters(obs_flux_diffs,read_vars,
                           min_b_val=1e-10,min_f_val=1e-10):
    '''
    takes observed flux differences (shape of n_pixels,n_reads-1) 
    and returns a good starting guess for the f and b vectors

    inputs:
        obs_flux_diffs: observed count differences, units of electrons, shape (n_pixels,n_reads-1)
        read_vars: read variance per pixel, units of electrons^2, shape (n_pixels)
        min_b_val: minimum value the b vector elements can take (should not be less than 0)
        min_f_val: minimum value the f vector elements can take (should not be less than 0)

    outputs:
        b_vect_guess: data-estimate of the b vector
        fmax_guess: data-estimate of the f vector
        max_b_ind: the index of the maximum element in b_vect_guess (scaled to be equal to 1)
    '''

    obs_flux_diff_errs = np.sqrt((read_vars[:,None]*2)**2+np.maximum(obs_flux_diffs,0))

    #take median of data to get first f vect
    fmax_guess = np.maximum(np.nanmedian(obs_flux_diffs,axis=1),min_f_val)

    #scale the data to get b vect estimates
    scaled_fluxes = (obs_flux_diffs/fmax_guess[:,None])
    scaled_flux_errs = np.abs(scaled_fluxes/np.maximum(obs_flux_diffs,min_f_val)*obs_flux_diff_errs)
    scaled_flux_errs[scaled_flux_errs == 0] = np.inf

    #take weighted average to get starting b vect
    weights = np.power(scaled_flux_errs,-2)
    b_vect_guess_errs = np.power(np.sum(weights,axis=0),-0.5)
    b_vect_guess = np.maximum(np.sum(weights*scaled_fluxes,axis=0)/np.sum(weights,axis=0),min_b_val)

    #scale the b vector so that maximum element is set to 1
    max_b_ind = np.argmax(b_vect_guess)
    b_scale = b_vect_guess[max_b_ind]
    
    b_vect_guess /= b_scale
    b_vect_guess_errs /= b_scale
    fmax_guess *= b_scale

    #now use the b vect guess to get a better f vect guess, accounting for uncertainty
    scaled_fluxes = (obs_flux_diffs/b_vect_guess)
    scaled_flux_errs = np.sqrt(np.power(scaled_fluxes/np.maximum(obs_flux_diffs,min_f_val)*obs_flux_diff_errs,2)\
                                +np.power(scaled_fluxes/b_vect_guess*b_vect_guess_errs,2))
    scaled_flux_errs[scaled_flux_errs == 0] = np.inf
    weights = np.power(scaled_flux_errs,-2)
    # fmax_guess_errs = np.power(np.sum(weights,axis=1),-0.5)
    fmax_guess = np.maximum(np.sum(weights*scaled_fluxes,axis=1)/np.sum(weights,axis=1),min_f_val)

    return b_vect_guess,fmax_guess,max_b_ind


def measure_time_dep_fluxes_LINEAR(obs_fluxes,read_errs,
                                   n_max_repeat=1000,b_vect_change_tol=1e-6,
                                   min_b_val=1e-10,min_f_val=1e-10,
                                   rescale=False,true_b_vect=None,
                                   true_fluxes=None):
    
    '''
    takes observed counts (shape of n_pixels,n_reads) 
    and returns the best-fit vect b and vect f with assocaited covariance matrix

    inputs:
        obs_fluxes: observed counts, units of electrons, shape (n_pixels,n_reads)
        read_errs: read noise per pixel, units of electrons, shape (n_pixels)
        n_max_repeat: maximum number of times to update the f,b vectors before stopping
        b_vect_change_tol: stop updating if the b vect values have changed by less than this tolerance
        min_b_val: minimum value the b vector elements can take (should not be less than 0)
        min_f_val: minimum value the f vector elements can take (should not be less than 0)
        rescale: if True, then use the input true b vector to scale f and b vectors (which can be done arbitrarily)
        true_b_vect: if rescale is True, then use true_b_vect to rescale the output f and b vectors

    outputs:
        max_b_ind: the index of the maximum element in b_vect_guess (scaled to be equal to 1)
        curr_b_vect: the b vector mean
        curr_b_Vinv: the b vector inverse covariance matrix
        curr_b_V: the b vector covariance matrix
        f_max_means_given_b: the f means given the final b mean
        f_max_ivars_given_b: the f ivars given the final b mean
        comb_param_means: the combined f and b vector mean
        comb_param_Vinv: the combined f and b vector inverse covariance matrix
        comb_param_V: the combined f and b vector covariance matrix
    '''

    n_reads = obs_fluxes.shape[1]
    n_pixels = obs_fluxes.shape[0]
    read_vars = np.power(read_errs,2)

    obs_flux_diffs = np.diff(obs_fluxes,axis=1)
    b_vect_guess,fmax_guess,max_b_ind = first_guess_parameters(obs_flux_diffs,read_vars,
                                           min_b_val=min_b_val,min_f_val=min_f_val)

    #define read covariance matrix 
    #(i.e. 2*read_var on diagonal, -1*read_var on one-from-diagonal)
    read_Vs = np.zeros((n_pixels,n_reads-1,n_reads-1))
    for p_ind in range(n_pixels):
        read_Vs[p_ind,np.arange(n_reads-1),np.arange(n_reads-1)] = 2*read_vars[p_ind]
        for j in range(len(read_Vs[p_ind])-1):
            read_Vs[p_ind,j,j+1] = -read_vars[p_ind]
            read_Vs[p_ind,j+1,j] = -read_vars[p_ind]

    #define matrix and vector needed to 
    #keep max element of b at exactly 1
    mat_01 = np.eye(n_reads-1)
    mat_01[max_b_ind,max_b_ind] = 0
    vec_10 = np.zeros(n_reads-1)
    vec_10[max_b_ind] = 1

    #define vectors that will be updated during iteration
    curr_b_vect = np.copy(b_vect_guess)
    curr_f_max = np.copy(fmax_guess)
            
    curr_f_max = np.maximum(curr_f_max,min_f_val)
    curr_b_vect = np.maximum(curr_b_vect,min_b_val)

    #do some preallocating
    previous_b_vect = np.copy(curr_b_vect)
    
    model_derivs = np.zeros((*obs_flux_diffs.shape,n_pixels+n_reads-2))
    non_max_inds = np.ones(len(curr_b_vect)).astype(bool)
    non_max_inds[max_b_ind] = False
    non_max_inds = np.where(non_max_inds)[0]

    data_V = np.zeros((n_pixels,n_reads-1,n_reads-1))

    previous_b_vect = np.copy(curr_b_vect)
    f_max_means = np.copy(curr_f_max)
    b_vect_means = np.copy(curr_b_vect)

    comb_param_means = np.zeros(n_pixels+n_reads-2)    
    comb_param_means[:n_pixels] = f_max_means
    comb_param_means[n_pixels:] = b_vect_means[non_max_inds]

    for r_ind in range(n_max_repeat):
        curr_model = curr_f_max[:,None] * curr_b_vect[None,:]
        curr_diffs = obs_flux_diffs-curr_model
        
        curr_data_diffs = np.maximum(curr_model,0)
        
        for p_ind in range(n_pixels):
            data_V[p_ind] = np.diag(curr_data_diffs[p_ind])
        comb_V_inv = np.linalg.inv(data_V + read_Vs)

        #define the local derivatives given the current parameter guess
        for p_ind in range(n_pixels):
            model_derivs[p_ind,:,p_ind] = curr_b_vect
        for p_ind in range(len(non_max_inds)):
            model_derivs[:,non_max_inds[p_ind],n_pixels+p_ind] = curr_f_max

        #calculate update vector based on these derivatives
        design_dot_Vinv = np.einsum('nij,nik->njk',model_derivs,comb_V_inv)
        comb_param_Vinv = np.sum(np.einsum('nij,njk->nik',design_dot_Vinv,model_derivs),axis=0)
        param_changes = np.linalg.solve(comb_param_Vinv,np.sum(np.einsum('nij,nj->ni',design_dot_Vinv,curr_diffs),axis=0))
        f_max_changes = param_changes[:n_pixels]
        b_vect_changes = param_changes[n_pixels:]

        #update the f and b vector guesses
        f_max_means = curr_f_max + f_max_changes
        b_vect_means[:] = curr_b_vect[:] 
        b_vect_means[non_max_inds] += b_vect_changes
        
        comb_param_means[:n_pixels] = f_max_means
        comb_param_means[n_pixels:] = b_vect_means[non_max_inds]

        #update guess
        curr_f_max = np.maximum(f_max_means,min_f_val)
        curr_b_vect = np.maximum(b_vect_means,min_b_val)
        
        if (np.max(np.abs(b_vect_means-previous_b_vect)) < b_vect_change_tol) and (r_ind >= 2):
            #stop iterating if the b vector isn't changing much
            #typically hit this condition after <10 iterations
            break
        
        previous_b_vect[:] = b_vect_means[:]

    if rescale:
        '''
        due to arbitrary tradeoff between f and b vector values, 
        scale the values to be as close to true b as possible for fair comparisons
        '''

        
        # #measure best single mult to b to get good agreement
        # true_b = true_b_vect[non_max_inds]
        # comp_b = curr_b_vect[non_max_inds]
        # mult_var = 1/np.dot(np.dot(comp_b,comb_param_Vinv[n_pixels:,n_pixels:]),comp_b)
        # mult_mean = 1/(mult_var * np.dot(np.dot(comp_b,comb_param_Vinv[n_pixels:,n_pixels:]),true_b))

        # true_obs_fluxes_diffs = np.diff(true_obs_fluxes,axis=1)
        # #measure best single mult to f to get good agreement
        mult_var = 1/np.dot(np.dot(f_max_means,comb_param_Vinv[:n_pixels,:n_pixels]),f_max_means)
        mult_mean = mult_var * np.dot(np.dot(f_max_means,comb_param_Vinv[:n_pixels,:n_pixels]),true_fluxes)
        # print(mult_mean,mult_var**0.5,np.nanmedian(true_fluxes/f_max_means),np.nanmean(true_fluxes/f_max_means))
        # # mult_mean = mult_var * np.dot(np.dot(f_max_means,comb_param_Vinv[:n_pixels,:n_pixels]),obs_flux_diffs[:,max_b_ind])
        # # print(mult_mean,mult_var**0.5)
        # # mult_mean = mult_var * np.dot(np.dot(f_max_means,comb_param_Vinv[:n_pixels,:n_pixels]),
        # #                               true_obs_flux_diffs[:,max_b_ind]/true_b_vect[max_b_ind])
        # # print(mult_mean,mult_var**0.5)

        # n_mults = 1000
        # test_mults = np.linspace(0.5*mult_mean,2.0*mult_mean,n_mults)
        # test_mult_lnprobs = np.zeros(n_mults)
        # comb_param_V = np.linalg.inv(comb_param_Vinv)
        # print(mult_mean,mult_var**0.5)

        # curr_V = np.copy(comb_param_V)
        # curr_vect = np.copy(comb_param_means)
        # true_vect = np.zeros_like(curr_vect)
        # true_vect[:n_pixels] = true_fluxes
        # true_vect[n_pixels:] = true_b_vect[non_max_inds]
        # for mult_ind,test_mult in enumerate(test_mults):
        #     curr_V[:] = comb_param_V[:]
        #     curr_V[:n_pixels] *= test_mult**2
        #     curr_V[:,:n_pixels] *= test_mult**2
        #     curr_V[n_pixels:] /= test_mult**2
        #     curr_V[:,n_pixels:] /= test_mult**2
            
        #     curr_vect[:] = comb_param_means[:]
        #     curr_vect[:n_pixels] *= test_mult
        #     curr_vect[n_pixels:] /= test_mult

        #     curr_diff = curr_vect - true_vect

        #     chi2 = np.dot(curr_diff,np.dot(np.linalg.inv(curr_V),curr_diff))
        #     test_mult_lnprobs[mult_ind] = -0.5*(np.log(np.linalg.det(curr_V))+chi2)

        # test_mult_probs = np.exp(test_mult_lnprobs-test_mult_lnprobs.max())
        # test_mult_probs /= np.sum(test_mult_probs)
        
        # plt.plot(test_mults,test_mult_lnprobs)
        # plt.show()
        # plt.plot(test_mults,test_mult_probs)
        # plt.show()
        # mult_mean = test_mults[np.argmax(test_mult_lnprobs)]
        # print(mult_mean)        
        
        #then rescale the fluxes and b values
        curr_f_max *= mult_mean
        curr_b_vect /= mult_mean

        f_max_means *= mult_mean
        b_vect_means /= mult_mean

        comb_param_means[:n_pixels] = f_max_means
        comb_param_means[n_pixels:] = b_vect_means[non_max_inds]

        comb_param_V = np.linalg.inv(comb_param_Vinv)
        comb_param_V[:n_pixels] *= mult_mean**2
        comb_param_V[:,:n_pixels] *= mult_mean**2
        comb_param_V[n_pixels:] /= mult_mean**2
        comb_param_V[:,n_pixels:] /= mult_mean**2

        comb_param_Vinv = np.linalg.inv(comb_param_V)

        vec_10[max_b_ind] = 1/mult_mean

    else:
        #only invert matrix at the end
        comb_param_V = np.linalg.inv(comb_param_Vinv)

    #calculate the best f given the best b
    curr_data_diffs = np.maximum(f_max_means[:,None] * b_vect_means[None,:],0)
    for p_ind in range(n_pixels):
        data_V[p_ind] = np.diag(curr_data_diffs[p_ind])
    comb_V_inv = np.linalg.inv(data_V + read_Vs)

    b_dot_01 = np.dot(mat_01,b_vect_means)+vec_10
    f_max_ivars_given_b = np.einsum('i,ni->n',b_dot_01,np.einsum('nij,j->ni',comb_V_inv,b_dot_01))
    f_max_means_given_b = (1/f_max_ivars_given_b) * np.einsum('i,ni->n',b_dot_01,np.einsum('nij,nj->ni',comb_V_inv,obs_flux_diffs))
            
    curr_b_V = np.eye(len(b_vect_means))*1e-20
    curr_b_Vinv = np.eye(len(b_vect_means))*1e20
    for j in range(len(non_max_inds)):
        curr_b_V[non_max_inds[j],non_max_inds] = comb_param_V[n_pixels+j,n_pixels:]
        curr_b_Vinv[non_max_inds[j],non_max_inds] = comb_param_Vinv[n_pixels+j,n_pixels:]
    curr_b_Vinv[max_b_ind] = 0
    curr_b_Vinv[:,max_b_ind] = 0
    curr_b_V[max_b_ind] = np.inf
    curr_b_V[:,max_b_ind] = np.inf
        
    return max_b_ind,b_vect_means,curr_b_Vinv,curr_b_V,\
                f_max_means_given_b,f_max_ivars_given_b,\
                comb_param_means,comb_param_Vinv,comb_param_V




def measure_time_dep_fluxes_GIBBS(obs_fluxes,read_errs,
                                   n_samples=1000,b_vect_change_tol=1e-6,
                                   min_b_val=1e-10,min_f_val=1e-10,
                                   rescale=False,true_b_vect=None,
                                   true_fluxes=None,
                                   use_linear_first_guess=False,
                                   verbose=True,return_samples=True):
    
    '''
    takes observed counts (shape of n_pixels,n_reads) 
    and returns the best-fit vect b and vect f with assocaited covariance matrix

    inputs:
        obs_fluxes: observed counts, units of electrons, shape (n_pixels,n_reads)
        read_errs: read noise per pixel, units of electrons, shape (n_pixels)
        n_samples: number of Gibbs samples to draw from the posterior (f,b | data) distribution
        b_vect_change_tol: stop updating if the b vect values have changed by less than this tolerance
        min_b_val: minimum value the b vector elements can take (should not be less than 0)
        min_f_val: minimum value the f vector elements can take (should not be less than 0)
        rescale: if True, then use the input true b vector to scale f and b vectors (which can be done arbitrarily)
        true_b_vect: if rescale is True, then use true_b_vect to rescale the output f and b vectors
        use_linear_first_guess: if True, then use the linearized method to define the first guess f and b, \
                                otherwise use first_guess_parameters function
        verbose: if True, then use tqdm to print progress bad of generating Gibbs samples
        return_samples: if True, then return the Gibbs samples of (f,b | data)
        
    outputs:
        max_b_ind: the index of the maximum element in b_vect_guess (scaled to be equal to 1)
        b_vect_means: the b vector mean
        curr_b_Vinv: the b vector inverse covariance matrix
        curr_b_V: the b vector covariance matrix
        f_max_means: the f means given the final b mean
        f_max_ivars: the f ivars given the final b mean
        comb_param_means: the combined f and b vector mean
        comb_param_Vinv: the combined f and b vector inverse covariance matrix
        comb_param_V: the combined f and b vector covariance matrix
        comb_param_samps: (only returned if return_samples == True), 

    '''

    n_reads = obs_fluxes.shape[1]
    n_pixels = obs_fluxes.shape[0]
    read_vars = np.power(read_errs,2)

    obs_flux_diffs = np.diff(obs_fluxes,axis=1)
    if use_linear_first_guess:
        #use the linearized method to get the first guess of f and b
        #which should put the starting guess very near the MAP
        max_b_ind,b_vect_means,curr_b_Vinv,curr_b_V,\
        f_max_means_given_b,f_max_ivars_given_b,\
        comb_param_means,comb_param_Vinv,comb_param_V = measure_time_dep_fluxes_LINEAR(obs_fluxes,read_errs,
                           b_vect_change_tol=b_vect_change_tol,
                           min_b_val=min_b_val,min_f_val=min_f_val,
                           rescale=rescale,true_b_vect=true_b_vect,
                           true_fluxes=true_fluxes)
        b_vect_guess,fmax_guess = b_vect_means,f_max_means_given_b
    else:
        #use the simple starting guess from first_guess_parameters
        b_vect_guess,fmax_guess,max_b_ind = first_guess_parameters(obs_flux_diffs,read_vars,
                                               min_b_val=min_b_val,min_f_val=min_f_val)

    #define read covariance matrix 
    #(i.e. 2*read_var on diagonal, -1*read_var on one-from-diagonal)
    read_Vs = np.zeros((n_pixels,n_reads-1,n_reads-1))
    for p_ind in range(n_pixels):
        read_Vs[p_ind,np.arange(n_reads-1),np.arange(n_reads-1)] = 2*read_vars[p_ind]
        for j in range(len(read_Vs[p_ind])-1):
            read_Vs[p_ind,j,j+1] = -read_vars[p_ind]
            read_Vs[p_ind,j+1,j] = -read_vars[p_ind]

    #define matrix and vector needed to 
    #keep max element of b at exactly 1
    mat_01 = np.eye(n_reads-1)
    mat_01[max_b_ind,max_b_ind] = 0
    vec_10 = np.zeros(n_reads-1)
    vec_10[max_b_ind] = 1

    #define vectors that will be updated during iteration
    curr_b_vect = np.copy(b_vect_guess)
    curr_f_max = np.copy(fmax_guess)
            
    curr_f_max = np.maximum(curr_f_max,min_f_val)
    curr_b_vect = np.maximum(curr_b_vect,min_b_val)

    #only use the non-max-b indices in the fitting
    non_max_inds = np.ones(len(curr_b_vect)).astype(bool)
    non_max_inds[max_b_ind] = False
    non_max_inds = np.where(non_max_inds)[0]

    #preallocate
    data_V = np.zeros((n_pixels,n_reads-1,n_reads-1))
    previous_b_vect = np.copy(curr_b_vect)
    comb_param_samps = np.zeros((n_samples,n_pixels+n_reads-2))

    if verbose:
        print(f'Using Gibbs sampling to generate {n_samples} samples')
        
    for s_ind,_ in enumerate(tqdm(np.arange(n_samples),total=n_samples,disable=not(verbose))):
        #define covariance matrix
        curr_model = curr_f_max[:,None] * curr_b_vect[None,:]        
        curr_data_diffs = np.maximum(curr_model,0)
        
        for p_ind in range(n_pixels):
            data_V[p_ind] = np.diag(curr_data_diffs[p_ind])
        comb_V_inv = np.linalg.inv(data_V + read_Vs)

        #calculate f given b
        b_dot_01 = np.dot(mat_01,curr_b_vect)+vec_10
        f_max_ivars_given_b = np.einsum('i,ni->n',b_dot_01,np.einsum('nij,j->ni',comb_V_inv,b_dot_01))
        f_max_means_given_b = (1/f_max_ivars_given_b) * np.einsum('i,ni->n',b_dot_01,\
                                                                  np.einsum('nij,nj->ni',\
                                                                            comb_V_inv,obs_flux_diffs))

        #draw new f vector
        curr_f_max = f_max_means_given_b+np.random.randn(n_pixels)*np.power(f_max_ivars_given_b,-0.5)
        comb_param_samps[s_ind,:n_pixels] = curr_f_max
        curr_f_max = np.maximum(curr_f_max,min_f_val)

        #re-define covariance matrix
        curr_model = curr_f_max[:,None] * curr_b_vect[None,:]        
        curr_data_diffs = np.maximum(curr_model,0)
        
        for p_ind in range(n_pixels):
            data_V[p_ind] = np.diag(curr_data_diffs[p_ind])
        comb_V_inv = np.linalg.inv(data_V + read_Vs)

        #calculate b given f
        smaller_Vinv = comb_V_inv[:,non_max_inds][:,:,non_max_inds]
        b_Vinv_given_f = np.sum(np.power(curr_f_max,2)[:,None,None]*smaller_Vinv,axis=0)
        b_mean_given_f = np.linalg.solve(b_Vinv_given_f,
                            np.sum(np.einsum('nij,nj->ni',curr_f_max[:,None,None]*smaller_Vinv,\
                                             obs_flux_diffs[:,non_max_inds]),axis=0))
        b_V_given_f = np.linalg.inv(b_Vinv_given_f)

        #draw new b vector
        curr_L = np.linalg.cholesky(b_V_given_f)
        curr_b_draw = np.dot(curr_L,np.random.randn(len(b_mean_given_f)))+b_mean_given_f
        comb_param_samps[s_ind,n_pixels:] = curr_b_draw
        curr_b_draw = np.maximum(curr_b_draw,min_b_val)

        curr_b_vect[non_max_inds] = curr_b_draw
        

    #use posterior samples to define posterior mean and covariance
    comb_param_means = np.nanmean(comb_param_samps,axis=0)
    comb_param_V = np.cov(comb_param_samps,rowvar=False)
    comb_param_Vinv = np.linalg.inv(comb_param_V)

    f_max_means = comb_param_means[:n_pixels]
    b_vect_means = curr_b_vect
    b_vect_means[non_max_inds] = comb_param_means[n_pixels:]
    if verbose:
        print('Done')

    if rescale:
        '''
        due to arbitrary tradeoff between f and b vector values, 
        scale the values to be as close to true b as possible for fair comparisons
        '''
        
        # # #measure best single mult to b to get good agreement
        # true_b = true_b_vect[non_max_inds]
        # comp_b = b_vect_means[non_max_inds]
        # mult_var = 1/np.dot(np.dot(comp_b,comb_param_Vinv[n_pixels:,n_pixels:]),comp_b)
        # mult_mean = 1/(mult_var * np.dot(np.dot(comp_b,comb_param_Vinv[n_pixels:,n_pixels:]),true_b))
        # # print(mult_mean,mult_var**0.5)

        # #measure best single mult to f to get good agreement
        mult_var = 1/np.dot(np.dot(f_max_means,comb_param_Vinv[:n_pixels,:n_pixels]),f_max_means)
        mult_mean = mult_var * np.dot(np.dot(f_max_means,comb_param_Vinv[:n_pixels,:n_pixels]),true_fluxes)
        
        #then rescale the fluxes and b values
        f_max_means *= mult_mean
        b_vect_means /= mult_mean

        comb_param_means[:n_pixels] = f_max_means
        comb_param_means[n_pixels:] = b_vect_means[non_max_inds]

        comb_param_samps[:,:n_pixels] *= mult_mean
        comb_param_samps[:,n_pixels:] /= mult_mean

        comb_param_V = np.linalg.inv(comb_param_Vinv)
        comb_param_V[:n_pixels] *= mult_mean**2
        comb_param_V[:,:n_pixels] *= mult_mean**2
        comb_param_V[n_pixels:] /= mult_mean**2
        comb_param_V[:,n_pixels:] /= mult_mean**2

        comb_param_Vinv = np.linalg.inv(comb_param_V)
            
    f_max_vars = np.array(np.diag(comb_param_V[:n_pixels,:n_pixels]))
    f_max_ivars = 1/f_max_vars

    curr_data_diffs = np.maximum(f_max_means[:,None] * b_vect_means[None,:],0)
    for p_ind in range(n_pixels):
        data_V[p_ind] = np.diag(curr_data_diffs[p_ind])
    comb_V_inv = np.linalg.inv(data_V + read_Vs)

    #calculate the best f given the best b
    b_dot_01 = np.dot(mat_01,b_vect_means)+vec_10
    f_max_ivars_given_b = np.einsum('i,ni->n',b_dot_01,np.einsum('nij,j->ni',comb_V_inv,b_dot_01))
    f_max_means_given_b = (1/f_max_ivars_given_b) * np.einsum('i,ni->n',b_dot_01,np.einsum('nij,nj->ni',comb_V_inv,obs_flux_diffs))

    f_max_means = f_max_means_given_b
    f_max_ivars = f_max_ivars_given_b
    
    curr_b_V = np.eye(len(b_vect_means))*1e-20
    curr_b_Vinv = np.eye(len(b_vect_means))*1e20
    for j in range(len(non_max_inds)):
        curr_b_V[non_max_inds[j],non_max_inds] = comb_param_V[n_pixels+j,n_pixels:]
        curr_b_Vinv[non_max_inds[j],non_max_inds] = comb_param_Vinv[n_pixels+j,n_pixels:]
    curr_b_Vinv[max_b_ind] = 0
    curr_b_Vinv[:,max_b_ind] = 0
    curr_b_V[max_b_ind] = np.inf
    curr_b_V[:,max_b_ind] = np.inf

    if return_samples:
        return max_b_ind,b_vect_means,curr_b_Vinv,curr_b_V,\
                    f_max_means,f_max_ivars,\
                    comb_param_means,comb_param_Vinv,comb_param_V,\
                    comb_param_samps
    else:
        return max_b_ind,b_vect_means,curr_b_Vinv,curr_b_V,\
                    f_max_means,f_max_ivars_given_b,\
                    comb_param_means,comb_param_Vinv,comb_param_V


def generate_data(read_errs,true_fluxes,true_b_vect):
    '''
    given true fluxes, b vectors, and read noise per pixel, sample from
    appropriate Poisson (for true counts) and Gaussian (for read noise) distributions
    to generate synthetic data

    inputs:
        read_errs: read noise per pixel, units of electrons, shape (n_pixels)
        true_fluxes: flux per pixel, units of electrons, shape (n_pixels)
        true_b_vect: number > 0 that is multiplicative factor to fluxes (n_reads-1)

    outputs:
        obs_fluxes: observed counts, units of electrons, shape (n_pixels,n_reads)
    '''

    n_reads = len(true_b_vect)+1
    n_pixels = len(read_errs)

    obs_fluxes = np.zeros((n_pixels,n_reads))

    true_count_rates = true_fluxes[:,None] * true_b_vect[None,:]
    obs_fluxes = np.zeros((n_pixels,n_reads))
    for p_ind in range(n_pixels):
        #poisson noise from arriving photons
        obs_fluxes[p_ind,1:] = np.cumsum(np.random.poisson(true_count_rates[p_ind]))
    true_obs_fluxes = np.copy(obs_fluxes)
    #add read noise
    obs_fluxes += np.random.randn(*obs_fluxes.shape)*read_errs[:,None]

    return obs_fluxes



