function MALA(likelihood_and_derivative_calculator,
              protein_at_observations,
              measurement_variance,
              number_of_samples,
              initial_position,
              step_size,
              proposal_covariance=I,
              thinning_rate=1,
              known_parameter_dict=Dict())

    mean_protein = mean(protein_at_observations[:,2])
    # initialise the covariance proposal matrix
    number_of_parameters = length(initial_position) - length(keys(known_parameter_dict))
    known_parameter_indices = [Int(known_parameter_dict[i][1]) for i in keys(known_parameter_dict)]
    known_parameter_values = [known_parameter_dict[i][2] for i in keys(known_parameter_dict)]
    unknown_parameter_indices = [i for i in 1:length(initial_position) if i ∉ known_parameter_indices]

    # check if default value is used, and set to q x q identity
    if proposal_covariance == I
        identity = true
    else
        identity = false
        proposal_cholesky = cholesky(proposal_covariance + 1e-8*I).L
    end

    proposal_covariance_inverse = inv(proposal_covariance)

    # initialise samples matrix and acceptance ratio counter
    accepted_moves = 0
    mcmc_samples = zeros(number_of_samples,number_of_parameters)
    mcmc_samples[1,:] = initial_position[unknown_parameter_indices]
    number_of_iterations = number_of_samples*thinning_rate

    # set LAP parameters
    k = 1
    c0 = 1.0
    c1 = log(10)/log(number_of_samples/5)

    # initial markov chain
    current_position = copy(initial_position)
    current_log_likelihood, current_log_likelihood_gradient = likelihood_and_derivative_calculator(protein_at_observations,
                                                                                                   current_position,
                                                                                                   mean_protein,
                                                                                                   measurement_variance)
    for iteration_index in 2:number_of_iterations
        # progress measure
        if iteration_index%(number_of_iterations//10)==0
            println(string("Progress: ",100*(iteration_index/number_of_iterations),'%'))
        end #if

        proposal = zeros(length(initial_position))
        if identity
            proposal[unknown_parameter_indices] .= ( current_position[unknown_parameter_indices] .+
                                                    step_size.*current_log_likelihood_gradient[unknown_parameter_indices]./2 +
                                                    sqrt(step_size)*randn(number_of_parameters) )
        else
            proposal[unknown_parameter_indices] .= ( current_position[unknown_parameter_indices] .+
                                                    step_size.*proposal_covariance*(current_log_likelihood_gradient[unknown_parameter_indices]./2) .+
                                                    sqrt(step_size).*proposal_cholesky*randn(number_of_parameters) )
        end

        # compute transition probabilities for acceptance step
        # fix known parameters
        # import pdb; pdb.set_trace()
        if length(known_parameter_dict) > 0
            proposal[known_parameter_indices] .= copy(known_parameter_values)
        end #if

        proposal_log_likelihood, proposal_log_likelihood_gradient = likelihood_and_derivative_calculator(protein_at_observations,
                                                                                                         proposal,
                                                                                                         mean_protein,
                                                                                                         measurement_variance)

        # if any of the parameters were negative we get -inf for the log likelihood
        if proposal_log_likelihood == -Inf
            if iteration_index%thinning_rate == 0
                mcmc_samples[Int64(iteration_index/thinning_rate),:] .= current_position[unknown_parameter_indices]
            end

            # LAP stuff also needed here
            acceptance_probability = 0
            if iteration_index%k == 0 && iteration_index > 1
                gamma_1 = 1/(iteration_index^c1)
                gamma_2 = c0*gamma_1
                log_step_size_squared = log(step_size^2) + gamma_2*(acceptance_probability - 0.574)
                step_size = sqrt(exp(log_step_size_squared))
            end
            continue
        end

        forward_helper_variable = ( proposal[unknown_parameter_indices] .- current_position[unknown_parameter_indices] .-
                                    step_size.*proposal_covariance*(current_log_likelihood_gradient[unknown_parameter_indices]./2) )
        backward_helper_variable = ( current_position[unknown_parameter_indices] .- proposal[unknown_parameter_indices] .-
                                     step_size.*proposal_covariance*(proposal_log_likelihood_gradient[unknown_parameter_indices]./2) )
        transition_kernel_pdf_forward = (-transpose(forward_helper_variable)*proposal_covariance_inverse*forward_helper_variable) /(2*step_size)
        transition_kernel_pdf_backward = (-transpose(backward_helper_variable)*proposal_covariance_inverse*backward_helper_variable)/(2*step_size)

        # accept-reject step
        acceptance_probability = min(1,exp(proposal_log_likelihood - transition_kernel_pdf_forward - current_log_likelihood + transition_kernel_pdf_backward))
        # print(acceptance_probability)
        if rand() < acceptance_probability
            current_position .= proposal
            current_log_likelihood = proposal_log_likelihood
            current_log_likelihood_gradient .= proposal_log_likelihood_gradient
            accepted_moves += 1
        end

        if iteration_index%thinning_rate == 0
            mcmc_samples[Int64(iteration_index/thinning_rate),:] .= current_position[unknown_parameter_indices]
        end

        # LAP stuff
        if iteration_index%k == 0 && iteration_index > 1
            gamma_1 = 1/(iteration_index^c1)
            gamma_2 = c0*gamma_1
            log_step_size_squared = log(step_size^2) + gamma_2*(acceptance_probability - 0.574)
            step_size = sqrt(exp(log_step_size_squared))
        end
    end # for
    println(string("Acceptance ratio: ",accepted_moves/number_of_iterations))

    return mcmc_samples
end

"""
A function which gives a (hopefully) decent MALA output for a given dataset with known or
unknown parameters. If a previous output already exists, this will be used to create a
proposal covariance matrix, otherwise one will be constructed with a two step warm-up
process.
"""
function run_mala_for_dataset(data_filename,
                              protein_at_observations,
                              measurement_variance,
                              number_of_parameters,
                              known_parameter_dict,
                              step_size = 1,
                              number_of_chains = 8,
                              number_of_samples = 80000)
    # make sure all data starts from time "zero"
    for i in 1:size(protein_at_observations,1)
        protein_at_observations[i,1] -= protein_at_observations[1,1]
    end

    mean_protein = mean(protein_at_observations[:,2])

    loading_path = joinpath(dirname(pathof(DelayedKalmanFilter)),"../test/data")
    saving_path = joinpath(dirname(pathof(DelayedKalmanFilter)),"../test/output")

    # if we already have mcmc samples, we can use them to construct a covariance matrix to directly sample
    if ispath(joinpath(saving_path,string("final_parallel_mala_output_",data_filename)))
        println("Posterior samples already exist, sampling directly without warm up...")

        mala_output = load(joinpath(saving_path,string("final_parallel_mala_output_",data_filename)),"mcmc_samples")
        previous_number_of_chains = size(mala_output,1)
        previous_number_of_samples = size(mala_output,2)
        previous_number_of_parameters = size(mala_output,3)

        # construct proposal covariance matrix
        new_number_of_samples = previous_number_of_samples - Int64(floor(previous_number_of_samples/2))
        burn_in = Int64(floor(previous_number_of_samples/2))+1

        samples_with_burn_in = reshape(mala_output[:,burn_in:end,:],
                                       (new_number_of_samples*previous_number_of_chains,previous_number_of_parameters))
        proposal_covariance = cov(samples_with_burn_in)

        # start from mean
        states = zeros((number_of_chains,7))
        states[:,[3,4]] = [log(log(2)/30),log(log(2)/90)]
        states[:,[1,2,5,6,7]] = mean(samples_with_burn_in,dims=1)

        # turn into array of arrays
        initial_states = [states[i,:] for i in 1:size(states,1)]

        # pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        # process_results = [ pool_of_processes.apply_async(MALA,
        #                                                   args=(protein_at_observations,
        #                                                         measurement_variance,
        #                                                         number_of_samples,
        #                                                         initial_state,
        #                                                         step_size,
        #                                                         proposal_covariance,
        #                                                         1,
        #                                                         known_parameters))
        #                     for initial_state in initial_states ]
        # ## Let the pool know that these are all so that the pool will exit afterwards
        # # this is necessary to prevent memory overflows.
        # pool_of_processes.close()
        #
        # array_of_chains = zeros(number_of_chains,number_of_samples,number_of_parameters)
        # for chain_index, process_result in enumerate(process_results):
        #     this_chain = process_result.get()
        #     array_of_chains[chain_index,:,:] = this_chain
        # pool_of_processes.join()
        array_of_chains = reshape(MALA(log_likelihood_and_derivative_with_prior_and_transformation,
                                       protein_at_observations,
                                       measurement_variance,
                                       number_of_samples,
                                       initial_states[1],
                                       step_size,
                                       proposal_covariance,
                                       1,
                                       known_parameter_dict),
                                  (1,number_of_samples,number_of_parameters))

        save(joinpath(saving_path,string("final_parallel_mala_output_",data_filename)),"mcmc_samples",array_of_chains)

    else
        # warm up chain
        println(string("New data set, initial warm up with ",string(Int64(floor(number_of_samples*0.3)))," samples..."))
        # Initialise by Latin Hypercube sampling
        println("Latin Hypercube sampling initial positions...")
        plan, _ = LHCoptim(20,number_of_parameters,1000)
        scaled_plan = scaleLHC(plan,[(minimum(protein_at_observations[:,2])/2,2*mean(protein_at_observations[:,2])),
                                     (2.5,5.5),
                                     (log(0.1),log(100.)),
                                     (log(0.1),log(35.)),
                                     (5.,35.)])
        # sample subset - one for each chain
        sampling_indices = sample(1:20,number_of_chains;replace=false)
        scaled_plan = scaled_plan[sampling_indices,:]

        states = zeros(number_of_chains,7)
        if number_of_chains > 1
            states[:,[3,4]] .= [log(log(2)/30),log(log(2)/90)]
        else
            states[:,[3,4]] = [log(log(2)/30),log(log(2)/90)]
        end
        states[:,[1,2,5,6,7]] .= scaled_plan
        # turn into array of arrays
        initial_states = [states[i,:] for i in 1:size(states,1)]

        println(string("Warming up with ",string(Int64(0.3*number_of_samples))," samples..."))
        initial_burnin_number_of_samples = Int64(0.3*number_of_samples)

        # pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        # process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
        #                                                   args=(protein_at_observations,
        #                                                         measurement_variance,
        #                                                         initial_burnin_number_of_samples,
        #                                                         initial_state,
        #                                                         step_size,
        #                                                         np.power(np.diag([2*mean_protein,4,9,8,39]),2),# initial variances are width of prior squared
        #                                                         1, # thinning rate
        #                                                         known_parameters))
        #                     for initial_state in initial_states ]
        # ## Let the pool know that these are all so that the pool will exit afterwards
        # # this is necessary to prevent memory overflows.
        # pool_of_processes.close()
        #
        # array_of_chains = zeros(number_of_chains,initial_burnin_number_of_samples,number_of_parameters)
        # for chain_index, process_result in enumerate(process_results):
        #     this_chain = process_result.get()
        #     array_of_chains[chain_index,:,:] = this_chain
        # pool_of_processes.join()

        proposal_covariance = diagm([2*mean_protein,4,9,8,39]).^2

        array_of_chains = reshape(MALA(log_likelihood_and_derivative_with_prior_and_transformation,
                                       protein_at_observations,
                                       measurement_variance,
                                       initial_burnin_number_of_samples,
                                       initial_states[1],
                                       step_size,
                                       proposal_covariance,
                                       1,
                                       known_parameter_dict),
                                  (1,initial_burnin_number_of_samples,number_of_parameters))

        save(joinpath(saving_path,string("first_parallel_mala_output_",data_filename)),"mcmc_samples",array_of_chains)

        println(string("Second warm up with ",string(Int64(number_of_samples*0.7))," samples..."))
        second_burnin_number_of_samples = Int64(0.7*number_of_samples)

        # construct proposal covariance matrix
        new_number_of_samples = initial_burnin_number_of_samples - Int64(floor(initial_burnin_number_of_samples/2))
        burn_in = Int64(floor(initial_burnin_number_of_samples/2))+1
        samples_with_burn_in = reshape(array_of_chains[:,burn_in:end,:],
                                       (new_number_of_samples*number_of_chains,number_of_parameters))
        proposal_covariance = cov(samples_with_burn_in)

        # make new initial states
        println("Latin Hypercube sampling initial positions...")
        plan, _ = LHCoptim(20,number_of_parameters,1000)
        scaled_plan = scaleLHC(plan,[(minimum(protein_at_observations[:,2])/2,2*mean(protein_at_observations[:,2])),
                                     (2.5,5.5),
                                     (log(0.1),log(100.)),
                                     (log(0.1),log(35.)),
                                     (5.,35.)])
        # sample subset - one for each chain
        sampling_indices = sample(1:20,number_of_chains;replace=false)
        scaled_plan = scaled_plan[sampling_indices,:]

        states = zeros(number_of_chains,7)
        if number_of_chains > 1
            states[:,[3,4]] .= [log(log(2)/30),log(log(2)/90)]
        else
            states[:,[3,4]] = [log(log(2)/30),log(log(2)/90)]
        end
        states[:,[1,2,5,6,7]] .= scaled_plan
        # turn into array of arrays
        initial_states = [states[i,:] for i in 1:size(states,1)]

        # pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        # process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
        #                                                   args=(protein_at_observations,
        #                                                         measurement_variance,
        #                                                         second_burnin_number_of_samples,
        #                                                         initial_state,
        #                                                         step_size,
        #                                                         proposal_covariance,
        #                                                         1, # thinning rate
        #                                                         known_parameters))
        #                     for initial_state in initial_states ]
        # ## Let the pool know that these are all finished so that the pool will exit afterwards
        # # this is necessary to prevent memory overflows.
        # pool_of_processes.close()
        #
        # array_of_chains = zeros(number_of_chains,second_burnin_number_of_samples,number_of_parameters)
        # for chain_index, process_result in enumerate(process_results):
        #     this_chain = process_result.get()
        #     array_of_chains[chain_index,:,:] = this_chain
        # pool_of_processes.join()

        array_of_chains = reshape(MALA(log_likelihood_and_derivative_with_prior_and_transformation,
                                       protein_at_observations,
                                       measurement_variance,
                                       second_burnin_number_of_samples,
                                       initial_states[1],
                                       step_size,
                                       proposal_covariance,
                                       1,
                                       known_parameter_dict),
                                  (1,second_burnin_number_of_samples,number_of_parameters))

        save(joinpath(saving_path,string("second_parallel_mala_output_",data_filename)),"mcmc_samples",array_of_chains)

        # sample directly
        println("Now sampling directly...")
        new_number_of_samples = second_burnin_number_of_samples - Int64(floor(second_burnin_number_of_samples/2))
        burn_in = Int64(floor(second_burnin_number_of_samples/2))+1
        samples_with_burn_in = reshape(array_of_chains[:,burn_in:end,:],
                                       (new_number_of_samples*number_of_chains,number_of_parameters))
        proposal_covariance = cov(samples_with_burn_in)

        # make new initial states
        println("Latin Hypercube sampling initial positions...")
        plan, _ = LHCoptim(20,number_of_parameters,1000)
        scaled_plan = scaleLHC(plan,[(minimum(protein_at_observations[:,2])/2,2*mean(protein_at_observations[:,2])),
                                     (2.5,5.5),
                                     (log(0.1),log(100.)),
                                     (log(0.1),log(35.)),
                                     (5.,35.)])
        # sample subset - one for each chain
        sampling_indices = sample(1:20,number_of_chains;replace=false)
        scaled_plan = scaled_plan[sampling_indices,:]

        states = zeros(number_of_chains,7)
        if number_of_chains > 1
            states[:,[3,4]] .= [log(log(2)/30),log(log(2)/90)]
        else
            states[:,[3,4]] = [log(log(2)/30),log(log(2)/90)]
        end
        states[:,[1,2,5,6,7]] .= scaled_plan
        # turn into array of arrays
        initial_states = [states[i,:] for i in 1:size(states,1)]

        # pool_of_processes = mp_pool.ThreadPool(processes = number_of_chains)
        # process_results = [ pool_of_processes.apply_async(hes_inference.kalman_mala,
        #                                                   args=(protein_at_observations,
        #                                                         measurement_variance,
        #                                                         number_of_samples,
        #                                                         initial_state,
        #                                                         step_size,
        #                                                         proposal_covariance,
        #                                                         1, # thinning rate
        #                                                         known_parameters))
        #                     for initial_state in initial_states ]
        # ## Let the pool know that these are all finished so that the pool will exit afterwards
        # # this is necessary to prevent memory overflows.
        # pool_of_processes.close()
        #
        # array_of_chains = np.zeros((number_of_chains,number_of_samples,number_of_parameters))
        # for chain_index, process_result in enumerate(process_results):
        #     this_chain = process_result.get()
        #     array_of_chains[chain_index,:,:] = this_chain
        # pool_of_processes.join()

        array_of_chains = reshape(MALA(log_likelihood_and_derivative_with_prior_and_transformation,
                                       protein_at_observations,
                                       measurement_variance,
                                       number_of_samples,
                                       initial_states[1],
                                       step_size,
                                       proposal_covariance,
                                       1,
                                       known_parameter_dict),
                                  (1,number_of_samples,number_of_parameters))

        save(joinpath(saving_path,string("final_parallel_mala_output_",data_filename)),"mcmc_samples",array_of_chains)
    end #if-else
end
