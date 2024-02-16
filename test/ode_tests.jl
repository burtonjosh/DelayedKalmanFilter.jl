using DelayedKalmanFilter: calculate_steady_state_of_ode
using DelimitedFiles
using Test

@testset "ODE tests" begin
  steady_state = calculate_steady_state_of_ode([10000.0, 5.0, log(2) / 30, log(2) / 90, 1.0, 1.0, 29.0])

  @test length(steady_state) == 2
  @test steady_state[1] ≈ 41.418588484100354
  @test steady_state[2] ≈ 5377.565839809256
end
