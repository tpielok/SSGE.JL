import SSGE
using Test

using Distributions
using LinearAlgebra
using ForwardDiff
using Distances
import Random

@testset "SSGEstimator" begin
    F = Float32
    test_dist = MvNormal([F(2),F(2)], Symmetric([F(1) F(0.3); F(0.3) F(1)]))
    
    test_log_pdf(x) = log.(pdf(test_dist, x))
    # Analytic solution
    ∇test_log_pdf(x) = ForwardDiff.gradient(test_log_pdf, x)
    
    Random.seed!(1337)
    num_samples = 100
    x_samples = rand(test_dist, num_samples)
    num_eig_fun = 3
    σ = F(1000.0)

    g = SSGE.SSGEstimator(x_samples, num_eig_fun, SSGE.SqExp(σ));     

    num_tests = 100
    x_tests = rand(test_dist, num_tests)
    
    @test mean([norm(∇test_log_pdf(x_tests[:, i]) - g(x_tests[:, i])) for i in 1:num_tests]) < 0.2

    g_tuned = SSGE.SSGEstimator(x_samples, F(0.95));     
    @test mean([norm(∇test_log_pdf(x_tests[:, i]) - g_tuned(x_tests[:, i])) for i in 1:num_tests]) < 0.2

end