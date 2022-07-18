using Test

@testset "get_transformations" begin
    lowers = [-Inf, 4.0, -Inf, 4.0]
    uppers = [Inf, Inf, 17.0, 17.0]
    x = Float64[-Inf, -103, -5, 0, 5, 57, Inf]
    for (lower, upper) in zip(lowers, uppers)
        from_R, to_R = get_transformations(lower, upper)
        @test all((to_R∘from_R).(x) .≈ x)
    end
end

@testset "Bounds constructor" begin
    lowers = [-Inf, 4.0, -Inf, 4.0]
    uppers = [Inf, Inf, 17.0, 17.0]
    x = Float64[-Inf, -103, -5, 0, 5, 57, Inf]
    for (_lower, _upper) in zip(lowers, uppers)
        bounds = Bounds(_lower, _upper)
        println(@test bound(lower(bounds)) === _lower)
        @show _lower
        @show _upper
        println("-----")
        @test bound(upper(bounds)) === _upper
        _from_R = from_R(bounds)
        _to_R = to_R(bounds)
        @test all((_to_R∘_from_R).(x) .≈ x)
    end
end
