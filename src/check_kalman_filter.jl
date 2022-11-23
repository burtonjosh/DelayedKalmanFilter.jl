function check_filter()
    loading_path = string(dirname(@__FILE__), "/../test/data/")
    protein_at_observations =
        readdlm(string(loading_path, "kalman_filter_test_trace_observations.csv"), ',')
    model_parameters = [10000.0, 5.0, log(2) / 30, log(2) / 90, 1.0, 1.0, 29.0]
    measurement_variance = 10000.0
    system_state, distributions =
        kalman_filter(protein_at_observations, model_parameters, measurement_variance)

    return system_state, distributions
end
