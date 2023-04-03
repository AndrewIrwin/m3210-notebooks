# Helper functions for various chapters

"""
    forwardsub(L,b)

Solve the lower triangular linear system with matrix `L` and
right-hand side vector `b`.
"""
function forwardsub(L,b)
    n = size(L,1)
    x = zeros(n)
    x[1] = b[1]/L[1,1]
    for i in 2:n
        s = sum( L[i,j]*x[j] for j in 1:i-1 )
        x[i] = ( b[i] - s ) / L[i,i]
    end
    return x
end

"""
    backsub(U,b)

Solve the upper triangular linear system with matrix `U` and
right-hand side vector `b`.
"""
function backsub(U,b)
    n = size(U,1)
    x = zeros(n)
    x[n] = b[n]/U[n,n]
    for i in n-1:-1:1
        s = sum( U[i,j]*x[j] for j in i+1:n )
        x[i] = ( b[i] - s ) / U[i,i]
    end
    return x
end

"""
    lsnormal(A,b)

Solve a linear least-squares problem by the normal equations.
Returns the minimizer of ||b-Ax||.
"""
function lsnormal(A,b)
    N = A'*A;  z = A'*b;
    R = cholesky(N).U
    w = forwardsub(R',z)                   # solve R'z=c
    x = backsub(R,w)                       # solve Rx=z
    return x
end

"""
    levenberg(f,x₁[;maxiter,ftol,xtol])

Use Levenberg's quasi-Newton iteration to find a root of the system
`f` starting from `x₁`. Returns the history of root estimates 
as a vector of vectors.

The optional keyword parameters set the maximum number of iterations
and the stopping tolerance for values of `f` and changes in `x`.

"""
function levenberg(f,x₁;maxiter=40,ftol=1e-12,xtol=1e-12)
    x = [float(x₁)]
    yₖ = f(x₁)
    k = 1;  s = Inf;
    A = fdjac(f,x[k],yₖ)   # start with FD Jacobian
    jac_is_new = true

    λ = 10;
    while (norm(s) > xtol) && (norm(yₖ) > ftol)
        # Compute the proposed step.
        B = A'*A + λ*I
        z = A'*yₖ
        s = -(B\z)
        
        x̂ = x[k] + s
        ŷ = f(x̂)

        # Do we accept the result?
        if norm(ŷ) < norm(yₖ)    # accept
            λ = λ/10   # get closer to Newton
            # Broyden update of the Jacobian.
            A += (ŷ-yₖ-A*s)*(s'/(s'*s))
            jac_is_new = false
            
            push!(x,x̂)
            yₖ = ŷ
            k += 1
        else                       # don't accept
            # Get closer to gradient descent.
            λ = 4λ
            # Re-initialize the Jacobian if it's out of date.
            if !jac_is_new
                A = fdjac(f,x[k],yₖ)
                jac_is_new = true
            end
        end

        if k==maxiter
            @warn "Maximum number of iterations reached."
            break
        end
        
    end
    return x
end

"""
    fdjac(f,x₀[,y₀])

Compute a finite-difference approximation of the Jacobian matrix for
`f` at `x₀`, where `y₀`=`f(x₀)` may be given.
"""
function fdjac(f,x₀,y₀=f(x₀))
    δ = sqrt(eps())*max(norm(x₀),1)   # FD step size
    m,n = length(y₀),length(x₀)
    if n==1
        J = (f(x₀+δ) - y₀) / δ
    else
        J = zeros(m,n)
        x = copy(x₀)
        for j in 1:n
            x[j] += δ
            J[:,j] = (f(x) - y₀) / δ
            x[j] -= δ
        end
    end
    return J
end

"""
    euler(ivp,n)

Apply Euler's method to solve the given IVP using `n` time steps.
Returns a vector of times and a vector of solution values.
"""
function euler(ivp,n)
    # Time discretization.
    a,b = ivp.tspan
    h = (b-a)/n
    t = [ a + i*h for i in 0:n ]

    # Initial condition and output setup.
    u = fill(float(ivp.u0),n+1)

    # The time stepping iteration.
    for i in 1:n
        u[i+1] = u[i] + h*ivp.f(u[i],ivp.p,t[i])
    end
    return t,u
end

"""
    ie2(ivp,n)

Apply the Improved Euler method to solve the given IVP using `n`
time steps. Returns a vector of times and a vector of solution
values.
"""
function ie2(ivp,n)
    # Time discretization.
    a,b = ivp.tspan
    h = (b-a)/n
    t = [ a + i*h for i in 0:n ]

    # Initialize output.
    u = fill(float(ivp.u0),n+1)

    # Time stepping.
    for i in 1:n
        uhalf = u[i] + h/2*ivp.f(u[i],ivp.p,t[i]);
        u[i+1] = u[i] + h*ivp.f(uhalf,ivp.p,t[i]+h/2);
    end
    return t,u
end


"""
    rk4(ivp,n)

Apply the common Runge-Kutta 4th order method to solve the given 
IVP using `n` time steps. Returns a vector of times and a vector of
solution values.
"""
function rk4(ivp,n)
    # Time discretization.
    a,b = ivp.tspan
    h = (b-a)/n
    t = [ a + i*h for i in 0:n ]

    # Initialize output.
    u = fill(float(ivp.u0),n+1)

    # Time stepping.
    for i in 1:n
        k₁ = h*ivp.f( u[i],      ivp.p, t[i]     )
        k₂ = h*ivp.f( u[i]+k₁/2, ivp.p, t[i]+h/2 )
        k₃ = h*ivp.f( u[i]+k₂/2, ivp.p, t[i]+h/2 )
        k₄ = h*ivp.f( u[i]+k₃,   ivp.p, t[i]+h   )
        u[i+1] = u[i] + (k₁ + 2(k₂+k₃) + k₄)/6
    end
    return t,u
end


"""
    diffmat2(n,xspan)

Compute 2nd-order-accurate differentiation matrices on `n`+1 points
in the interval `xspan`. Returns a vector of nodes and the matrices
for the first and second derivatives.
"""
function diffmat2(n,xspan)
    a,b = xspan
    h = (b-a)/n
    x = [ a + i*h for i in 0:n ]   # nodes

    # Define most of Dₓ by its diagonals.
    dp = fill(0.5/h,n)        # superdiagonal
    dm = fill(-0.5/h,n)       # subdiagonal
    Dₓ = diagm(-1=>dm,1=>dp)

    # Fix first and last rows.
    Dₓ[1,1:3] = [-1.5,2,-0.5]/h
    Dₓ[n+1,n-1:n+1] = [0.5,-2,1.5]/h

    # Define most of Dₓₓ by its diagonals.
    d0 =  fill(-2/h^2,n+1)    # main diagonal
    dp =  ones(n)/h^2         # super- and subdiagonal
    Dₓₓ = diagm(-1=>dp,0=>d0,1=>dp)

    # Fix first and last rows.
    Dₓₓ[1,1:4] = [2,-5,4,-1]/h^2
    Dₓₓ[n+1,n-2:n+1] = [-1,4,-5,2]/h^2

    return x,Dₓ,Dₓₓ
end

"""
    diffcheb(n,xspan)

Compute Chebyshev differentiation matrices on `n`+1 points in the
interval `xspan`. Returns a vector of nodes and the matrices for the
first and second derivatives.
"""
function diffcheb(n,xspan)
    x = [ -cos( k*π/n ) for k in 0:n ]    # nodes in [-1,1]
    
    # Off-diagonal entries.
    c = [2; ones(n-1); 2];    # endpoint factors
    dij = (i,j) -> (-1)^(i+j)*c[i+1]/(c[j+1]*(x[i+1]-x[j+1]))
    Dₓ = [ dij(i,j) for i in 0:n, j in 0:n ]

    # Diagonal entries.
    Dₓ[isinf.(Dₓ)] .= 0         # fix divisions by zero on diagonal
    s = sum(Dₓ,dims=2)
    Dₓ -= diagm(s[:,1])         # "negative sum trick"

    # Transplant to [a,b].
    a,b = xspan
    x = @. a + (b-a)*(x+1)/2
    Dₓ = 2*Dₓ/(b-a)             # chain rule

    # Second derivative.
    Dₓₓ = Dₓ^2
    return x,Dₓ,Dₓₓ
end

"""
    bvplin(p,q,r,xspan,lval,rval,n)

Use finite differences to solve a linear boundary value problem.
The ODE is u''+`p`(x)u'+`q`(x)u = `r`(x) on the interval `xspan`,
with endpoint function values given as `lval` and `rval`. There will
be `n`+1 equally spaced nodes, including the endpoints.

Returns vectors of the nodes and the solution values.
"""
function bvplin(p,q,r,xspan,lval,rval,n)
    x,Dₓ,Dₓₓ = diffcheb(n,xspan)

    P = diagm(p.(x))
    Q = diagm(q.(x))
    L = Dₓₓ + P*Dₓ + Q     # ODE expressed at the nodes

    # Replace first and last rows using boundary conditions.
    z = zeros(1,n)
    A = [ [1 z]; L[2:n,:]; [z 1] ]
    b = [ lval; r.(x[2:n]); rval ]

    # Solve the system.
    u = A\b
    return x,u
end

"""
    bvp(ϕ,xspan,lval,lder,rval,rder,init)

Finite differences to solve a two-point boundary value problem with
ODE u'' = `ϕ`(x,u,u') for x in `xspan`, left boundary condition 
`g₁`(u,u')=0, and right boundary condition `g₂`(u,u')=0. The value 
`init` is an initial estimate for the values of the solution u at
equally spaced values of x, which also sets the number of nodes.
    
Returns vectors for the nodes and the values of u.
"""
function bvp(ϕ,xspan,g₁,g₂,init)
    n = length(init) - 1
    x,Dₓ,Dₓₓ = diffmat2(n,xspan)
    h = x[2]-x[1]

    function residual(u)
        # Residual of the ODE at the nodes. 
        du_dx = Dₓ*u                   # discrete u'
        d2u_dx2 = Dₓₓ*u                # discrete u''
        f = d2u_dx2 - ϕ.(x,u,du_dx)

        # Replace first and last values by boundary conditions.
        f[1] = g₁(u[1],du_dx[1])/h  # /h is a bit arbitrary; du_dx has a /h in it, so this is a useful scaling factor
        f[n+1] = g₂(u[n+1],du_dx[n+1])/h
        return f
    end
    
    u = levenberg(residual,init)
    return x,u[end]
end
