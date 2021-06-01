using DelayedKalmanFilter
using DelimitedFiles
using Test

@testset "ODE tests" begin
    steady_state = calculate_steady_state_of_ode(10000.,5.,log(2)/30,log(2)/90,1.,1.);

    @test length(steady_state) == 2
    @test steady_state[1] ≈ 41.41785876041631
    @test steady_state[2] ≈ 5377.800549410291
end #testset
