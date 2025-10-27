#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>

#pragma region solver

// Solves a tridiagonal system Ax = d using the Thomas algorithm
// a, b, c are the sub-diagonal, main diagonal, and super-diagonal of A
// d is the right-hand side vector 
std::vector<double> solveTridiagonal(const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d) {
    int n = b.size();
    std::vector<double> c_star(n, 0.0);
    std::vector<double> d_star(n, 0.0);
    std::vector<double> x(n, 0.0);

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (int i = 1; i < n; i++) {
        double m = b[i] - a[i] * c_star[i - 1];
        c_star[i] = c[i] / m;
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / m;
    }

    x[n - 1] = d_star[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = d_star[i] - c_star[i] * x[i + 1];
    }

    return x;
}

#pragma endregion

int main() {

    // Geometry and numerics
    const double L = 0.10;              // Length of the domain [m]
    const int    Nx = 10;               // Number of nodes [-]
    const double dx = L / (Nx - 1);     // Grid spacing [m]

    const double dt = 0.0001;                             // Timestep [s]
    const double t_max = 0.1;                              // Time interval [s]
    const int    t_iter = (int)std::round(t_max / dt);      // Number of timesteps [-]

    const int tot_iter = 200;       // Inner iterations per step [-]
    const int corr_iter = 2;        // PISO correctors per iteration [-]
    const double tol = 1e-8;        // Tolerance for the inner iterations [-]

    // Vapor properties (water vapor)
    const double Rv = 461.5;     // Gas constant [J/(kg K)]
    const double cp = 2010.0;    // Specific heat at constant pressure [J/(kg K)]
    const double mu = 1.3e-5;    // Dynamic viscosity [Pa s]
    const double k_cond = 0.028;      // Thermal conductivity W/(m K)

    // Fields
    std::vector<double> u(Nx + 1, 0.0), p(Nx, 50000.0), T(Nx, 380.0), rho(Nx, 0.0);
    std::vector<double> p_prime(Nx, 0.0);

    // Boundary conditions (Dirichlet p at outlet, T at both ends, u inlet)
    const double u_inlet = 0.0;        // Inlet velocity [m/s]
    const double u_outlet = 0.0;        // Outlet velocity [m/s]
    const double p_outlet = 50000.0;  // Outlet pressure [Pa]
    const double T_inlet = 390.0;    // Inlet temperature [K] (evaporator)
    const double T_outlet = 350.0;   // Outlet temperature [K] (condenser)

    auto eos_update = [&](std::vector<double>& rho_, const std::vector<double>& p_, const std::vector<double>& T_) {
    #pragma omp parallel
        for (int i = 0; i < Nx; i++) {
            double Ti = std::max(200.0, T_[i]);
            rho_[i] = std::max(1e-6, p_[i] / (Rv * Ti));
        }
    }; eos_update(rho, p, T);

    // Mass source
    std::vector<double> Sm(Nx, 0.0);
    Sm[1] = 1000.0; // mass source term at cell 1
    Sm[2] = 1000.0; // mass source term at cell 1
    Sm[7] = -1000.0; // mass sink term at cell 8
    Sm[8] = -1000.0; // mass sink term at cell 8

    // Momentum source
    std::vector<double> Su(Nx, 0.0);

    // Output
    std::ofstream fout("solution_vapor_PISO_thermal.txt");

    for (double it = 0; it < t_iter; it++) {

        printf("Time: %f / %f, Courant number: %f \n", dt * it, t_max, u_inlet * dt / dx);

        // PISO pressure correction loops
        double maxErr = 1.0;
        int iter = 0;

        while (iter<tot_iter && maxErr>tol) {

            // Momentum predictor
            // Backward Euler in time, central diffusion, pressure gradient explicit
            std::vector<double> a(Nx, 0.0), b(Nx, 2 * 4.0 / 3.0 * mu / dx), c(Nx, 0.0), d(Nx, 0.0);

            #pragma omp parallel
            for (int i = 1; i < Nx; i++) {



                printf("");
            }

            // Velocity BC: Dirichlet at left, zero-gradient at right
            b[0] = 1.0; c[0] = 0.0; d[0] = u_inlet;
            a[Nx - 1] = 0.0; b[Nx - 1] = 1.0; d[Nx - 1] = u_outlet;

            const double D = 4.0 / 3.0 * mu / dx;

            // Second node
            double rhie_chow_left = 1.0 / (b[0] + b[1]) / dx * (-2 * p[0] + 3 * p[1] - p[2]);
            double rhie_chow_right = 1.0 / (b[2] + b[1]) / dx * (p[0] - 3 * p[1] + 3 * p[2] - p[3]);

            double u_left_face = 0.5 * (u[0] + u[1]) + rhie_chow_left;
            double u_right_face = 0.5 * (u[1] + u[2]) + rhie_chow_right;

            if (u_left_face >= 0 && u_right_face >= 0) {

                a[1] = -u_left_face * rho[0] - D;
                c[1] = -D;
                b[1] = u_right_face * rho[1] + rho[1] * dx / dt + 2 * D;
                d[1] = -0.5 * (p[2] - p[0]) + rho[1] * u[1] * dx / dt + Sm[1] * dx;

            }
            else if (u_left_face >= 0 && u_right_face < 0) {

                a[1] = -u_left_face * rho[0] - D;
                c[1] = u_right_face * rho[2] - D;
                b[1] = rho[1] * dx / dt + 2 * D;
                d[1] = -0.5 * (p[2] - p[0]) + rho[1] * u[1] * dx / dt + Sm[1] * dx;

            }
            else if (u_left_face < 0 && u_right_face >= 0) {

                a[1] = -D;
                c[1] = -D;
                b[1] = (u_right_face - u_left_face) * rho[1] + rho[1] * dx / dt + 2 * D;
                d[1] = -0.5 * (p[2] - p[0]) + rho[1] * u[1] * dx / dt + Sm[1] * dx;

            }
            else if (u_left_face < 0 && u_right_face < 0) {

                a[1] = -D;
                c[1] = u_right_face * rho[2] - D;
                b[1] = -u_left_face * rho[1] + rho[1] * dx / dt + 2 * D;
                d[1] = -0.5 * (p[2] - p[0]) + rho[1] * u[1] * dx / dt + Sm[1] * dx;

            }

            // Second-to-last node
            rhie_chow_left = 1.0 / (b[Nx - 3] + b[Nx - 2]) / dx * (p[Nx - 4] - 3 * p[Nx - 3] + 2 * p_outlet);
            rhie_chow_right = 1.0 / (b[Nx - 1] + b[Nx - 2]) / dx * (p[Nx - 3] - 3 * p[Nx - 2] + 2 * p_outlet);

            u_left_face = 0.5 * (u[Nx - 3] + u[Nx - 2]) + rhie_chow_left;
            u_right_face = 0.5 * (u[Nx - 2] + u[Nx - 1]) + rhie_chow_right;

            if (u_left_face >= 0 && u_right_face >= 0) {

                a[Nx - 2] = -u_left_face * rho[Nx - 2] - D;
                c[Nx - 2] = -D;
                b[Nx - 2] = u_right_face * rho[Nx - 2] + rho[Nx - 2] * dx / dt + 2 * D;
                d[Nx - 2] = -0.5 * (p[Nx - 1] - p[Nx - 3]) + rho[Nx - 2] * u[Nx - 2] * dx / dt + Sm[Nx - 2] * dx;

            }
            else if (u_left_face >= 0 && u_right_face < 0) {

                a[Nx - 2] = -u_left_face * rho[Nx - 3] - D;
                c[Nx - 2] = u_right_face * rho[Nx - 1] - D;
                b[Nx - 2] = rho[Nx - 2] * dx / dt + 2 * D;
                d[Nx - 2] = -0.5 * (p[Nx - 1] - p[Nx - 3]) + rho[Nx - 2] * u[Nx - 2] * dx / dt + Sm[Nx - 2] * dx;

            }
            else if (u_left_face < 0 && u_right_face >= 0) {

                a[Nx - 2] = -D;
                c[Nx - 2] = -D;
                b[Nx - 2] = (u_right_face - u_left_face) * rho[Nx - 2] + rho[Nx - 2] * dx / dt + 2 * D;
                d[Nx - 2] = -0.5 * (p[Nx - 1] - p[Nx - 3]) + rho[Nx - 2] * u[Nx - 2] * dx / dt + Sm[Nx - 2] * dx;

            }
            else if (u_left_face < 0 && u_right_face < 0) {

                a[Nx - 2] = -D;
                c[Nx - 2] = u_right_face * rho[Nx - 1] - D;
                b[Nx - 2] = -u_left_face * rho[Nx - 2] + rho[Nx - 2] * dx / dt + 2 * D;
                d[Nx - 2] = -0.5 * (p[Nx - 1] - p[Nx - 3]) + rho[Nx - 2] * u[Nx - 2] * dx / dt + Sm[Nx - 2] * dx;

            }

            u = solveTridiagonal(a, b, c, d);

            for (int piso = 0; piso < corr_iter; piso++) {

                std::vector<double> aP(Nx, 0.0), bP(Nx, 0.0), cP(Nx, 0.0), dP(Nx, 0.0);

#pragma omp parallel
                for (int i = 2; i < Nx - 2; i++) {

                    const double source = Sm[i] * dx;
                    const double time_term = dx / dt;

                    const double rhie_chow_left = 1.0 / (b[i - 1] + b[i]) / dx * (p[i - 2] - 3 * p[i - 1] + 3 * p[i] - p[i + 1]);
                    const double rhie_chow_right = 1.0 / (b[i + 1] + b[i]) / dx * (p[i - 1] - 3 * p[i] + 3 * p[i + 1] - p[i + 2]);

                    const double u_left_face = 0.5 * (u[i - 1] + u[i]) + rhie_chow_left;
                    const double u_right_face = 0.5 * (u[i] + u[i + 1]) + rhie_chow_right;

                    aP[i] = -std::max(u_left_face, 0.0);
                    cP[i] = -std::max(-u_right_face, 0.0);
                    bP[i] = time_term + std::max(-u_left_face, 0.0) + std::max(u_right_face, 0.0);
                    dP[i] = time_term * rho[i] + source;

                    printf("");
                }

                // BCs for p'
                // Left: zero gradient correction
                bP[0] = 1.0; cP[0] = -1.0; dP[0] = 0.0;

                // Right: p' = 0 to pin pressure level
                bP[Nx - 1] = 1.0; aP[Nx - 1] = 0.0; dP[Nx - 1] = 0.0;

                // Second node
                double source = Sm[1] * dx;
                double time_term = dx / dt;

                double rhie_chow_left = 1.0 / (b[0] + b[1]) / dx * (-2 * p[0] + 3 * p[1] - p[2]);
                double rhie_chow_right = 1.0 / (b[2] + b[1]) / dx * (p[0] - 3 * p[1] + 3 * p[2] - p[3]);

                double u_left_face = 0.5 * (u[0] + u[1]) + rhie_chow_left;
                double u_right_face = 0.5 * (u[1] + u[2]) + rhie_chow_right;

                aP[1] = -std::max(u_left_face, 0.0);
                cP[1] = -std::max(-u_right_face, 0.0);
                bP[1] = time_term + std::max(-u_left_face, 0.0) + std::max(u_right_face, 0.0);
                dP[1] = time_term * rho[1] + source;

                // Second-to-last node
                source = Sm[Nx - 2] * dx;
                time_term = dx / dt;

                rhie_chow_left = 1.0 / (b[Nx - 3] + b[Nx - 2]) / dx * (p[Nx - 4] - 3 * p[Nx - 3] + 3 * p[Nx - 2] - p[Nx - 1]);
                rhie_chow_right = 1.0 / (b[Nx - 1] + b[Nx - 2]) / dx * (p[Nx - 3] - 3 * p[Nx - 2] + 2 * p_outlet);

                u_left_face = 0.5 * (u[Nx - 3] + u[Nx - 2]) + rhie_chow_left;
                u_right_face = 0.5 * (u[Nx - 2] + u[Nx - 1]) + rhie_chow_right;

                aP[Nx - 2] = -std::max(u_left_face, 0.0);
                cP[Nx - 2] = -std::max(-u_right_face, 0.0);
                bP[Nx - 2] = time_term + std::max(-u_left_face, 0.0) + std::max(u_right_face, 0.0);
                dP[Nx - 2] = time_term * rho[Nx - 2] + source;

                p_prime = solveTridiagonal(aP, bP, cP, dP);

                // Correct p and u
                for (int i = 0; i < Nx; i++) p[i] += p_prime[i];

                maxErr = 0.0;
                for (int i = 1; i < Nx - 1; i++) {
                    double u_prev = u[i];
                    u[i] = u[i] - (p_prime[i + 1] - p_prime[i - 1]) / (2.0 * dx * b[i]);
                    maxErr = std::max(maxErr, std::fabs(u[i] - u_prev));
                }

                // Velocity BC after correction
                u[0] = 0.0;
                u[Nx - 1] = 0.0;

                // Enforce outlet pressure
                p[Nx - 1] = p_outlet; p[0] = p[1];
            }

            iter++;
        }

        // Update density with new p,T
        eos_update(rho, p, T);

        //// --- Turbulence transport equations (1D implicit form) ---
        //const double sigma_k = 0.85;
        //const double sigma_omega = 0.5;
        //const double beta_star = 0.09;
        //const double beta = 0.075;
        //const double alpha = 5.0 / 9.0;

        //std::vector<double> aK(Nx, 0.0), bK(Nx, 0.0), cK(Nx, 0.0), dK(Nx, 0.0);
        //std::vector<double> aW(Nx, 0.0), bW(Nx, 0.0), cW(Nx, 0.0), dW(Nx, 0.0);

        //// --- Compute strain rate and production ---
        //std::vector<double> dudx(Nx, 0.0);
        //std::vector<double> Pk(Nx, 0.0);

        //for (int i = 1; i < Nx - 1; i++) {
        //    dudx[i] = (u[i + 1] - u[i - 1]) / (2.0 * dx);
        //    Pk[i] = mu_t[i] * pow(dudx[i], 2.0);
        //}

        //// --- k-equation ---
        //for (int i = 1; i < Nx - 1; i++) {
        //    double mu_eff = mu + mu_t[i];
        //    double Dw = mu_eff / (sigma_k * dx * dx);
        //    double De = mu_eff / (sigma_k * dx * dx);
        //           aK[i] = -Dw;
        //    cK[i] = -De;
        //    bK[i] = rho[i] / dt + Dw + De + beta_star * rho[i] * omega_turb[i];
        //    dK[i] = rho[i] / dt * k_turb[i] + Pk[i];
        //}
        //
        //// k BCs: constant initial values at the boundaries
        //bK[0] = 1.0; dK[0] = k_turb[0]; cK[0] = 0.0;
        //aK[Nx - 1] = 0.0; bK[Nx - 1] = 1.0; dK[Nx - 1] = k_turb[Nx - 1];

        //k_turb = solveTridiagonal(aK, bK, cK, dK);

        //// --- omega-equation ---
        //for (int i = 1; i < Nx - 1; i++) {
        //    double mu_eff = mu + mu_t[i];
        //    double Dw = mu_eff / (sigma_omega * dx * dx);
        //    double De = mu_eff / (sigma_omega * dx * dx);

        //    aW[i] = -Dw;
        //    cW[i] = -De;
        //    bW[i] = rho[i] / dt + Dw + De + beta * rho[i] * omega_turb[i];
        //    dW[i] = rho[i] / dt * omega_turb[i] + alpha * (omega_turb[i] / k_turb[i]) * Pk[i];
        //}
        //bW[0] = 1.0; dW[0] = omega_turb[0]; cW[0] = 0.0;
        //aW[Nx - 1] = 0.0; bW[Nx - 1] = 1.0; dW[Nx - 1] = omega_turb[Nx - 1];

        //omega_turb = solveTridiagonal(aW, bW, cW, dW);

        //// --- Update turbulent viscosity ---
        //for (int i = 0; i < Nx; i++) {
        //    double denom = std::max(omega_turb[i], 1e-6);
        //    mu_t[i] = rho[i] * k_turb[i] / denom;
        //    mu_t[i] = std::min(mu_t[i], 1000.0 * mu); // limiter
        //}

        // Energy equation for T (implicit), upwind convection, central diffusion
        std::vector<double> aT(Nx, 0.0), bT(Nx, 0.0), cT(Nx, 0.0), dT(Nx, 0.0);

#pragma omp parallel
        for (int i = 1; i < Nx - 1; i++) {
            double rhoCp = rho[i] * cp;
            double Pr_t = 0.9;
            double keff = k_cond/* + mu_t[i] * cp / Pr_t*/;
            double Dw = keff / (dx * dx);
            double De = keff / (dx * dx);

            // Upwind mass flux-like term F = rho*u; here in 1D use nodal u and rho
            double Fw = 0.5 * (rho[i - 1] * u[i - 1] + rho[i] * u[i]);
            double Fe = 0.5 * (rho[i] * u[i] + rho[i + 1] * u[i + 1]);

            double aw = std::max(Fw, 0.0) + Dw;
            double ae = std::max(-Fe, 0.0) + De;

            aT[i] = -aw;
            cT[i] = -ae;
            bT[i] = rhoCp / dt + aw + ae;
            dT[i] = rhoCp / dt * T_old[i];
        }
        // T BCs
        bT[0] = 1.0; dT[0] = T_inlet; cT[0] = 0.0;
        aT[Nx - 1] = 0.0; bT[Nx - 1] = 1.0; dT[Nx - 1] = T_outlet;

        T = solveTridiagonal(aT, bT, cT, dT);

        // Update density again after T change
        eos_update(rho, p, T);

        // Write last step profiles
        if (it == (t_iter - 1)) {
            for (int i = 0; i < Nx; i++) {
                fout << u[i] << ", ";
            }
        } fout << "\n\n";

        // Write last step profiles
        if (it == (t_iter - 1)) {
            for (int i = 0; i < Nx; i++) {
                fout << p[i] << ", ";
            }
        } fout << "\n\n";

        // Write last step profiles
        if (it == (t_iter - 1)) {
            for (int i = 0; i < Nx; i++) {
                // fout << T[i] << ", ";
            }
        }
    }

    return 0;
}
