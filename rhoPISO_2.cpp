#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <algorithm>

// =======================================================================
//
//                        [SOLVING ALGORITHMS]
//
// =======================================================================

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

    // =======================================================================
    //
    //                      [CONSTANTS AND VARIABLES]
    //
    // =======================================================================

    #pragma region constant_variables

    // Geometry and numerics
    const double L = 1.0;                       // Length of the domain [m]
    const int    N = 100;                       // Number of nodes [-]
    const double dz = L / (N - 1);              // Grid spacing [m]
    const double D_pipe = 0.1;                  // Pipe diameter [m], used only to estimate Reynolds number

    const double dt = 0.001;                           // Timestep [s]
    const double t_max = 1.0;                          // Time interval [s]
    const int    t_iter = (int)std::round(t_max / dt);      // Number of timesteps [-]

    const int tot_iter = 200;                   // Inner iterations per step [-]
    const int corr_iter = 2;                    // PISO correctors per iteration [-]
    const double tol = 1e-8;                    // Tolerance for the inner iterations [-]

    // Vapor properties (water vapor)
    const double Rv = 461.5;                    // Gas constant [J/(kg K)]
    const double cp = 2010.0;                   // Specific heat at constant pressure [J/(kg K)]
    const double mu = 1.3e-5;                   // Dynamic viscosity [Pa s]
    const double k_cond = 0.028;                // Thermal conductivity W/(m K)
    const double D = 4.0 / 3.0 * mu / dz;       // Diffusion coefficient [kg/(m^2 s)]

    // Fields
    std::vector<double> u(N, 0.01), p(N, 50000.0), T(N, 380.0), rho(N, 0.5);
    std::vector<double> T_old(N, 380.0), rho_old(N, 0.5), p_old(N, 50000.0);
    std::vector<double> p_prime(N, 0.0);

    // Boundary conditions (Dirichlet p at outlet, T at both ends, u inlet)
    const double u_inlet = 0.0;         // Inlet velocity [m/s]
    const double u_outlet = 0.0;        // Outlet velocity [m/s]
    const double p_outlet = 50000.0;    // Outlet pressure [Pa]
    const double T_inlet = 390.0;       // Inlet temperature [K] (evaporator)
    const double T_outlet = 350.0;      // Outlet temperature [K] (condenser)

    // Update equation of state
    auto eos_update = [&](std::vector<double>& rho_, const std::vector<double>& p_, const std::vector<double>& T_) {
    // #pragma omp parallel
        for (int i = 0; i < N; i++) {
            double Ti = std::max(200.0, T_[i]);
            rho_[i] = std::max(1e-6, p_[i] / (Rv * Ti));
        }
    }; eos_update(rho, p, T);

    // Mass source and sink definitions

    std::vector<double> Sm(N, 0.0);

    const double source_zone = 0.2;
    const double sink_zone = 0.2;

    const double source_nodes = std::floor(N * source_zone);
    const double sink_nodes = std::floor(N * sink_zone);

    for (int ix = 1; ix < N - 1; ++ix) {
        
        if (ix > 0 && ix <= source_nodes) Sm[ix] = 1.0;
        else if (ix >= (N - sink_nodes) && ix < (N - 1)) Sm[ix] = -1.0;

    }

    // Momentum source
    std::vector<double> Su(N, 0.0);

    // Turbulence constants for sodium vapor (SST model)
    const double I = 0.05;                              // Turbulence intensity (5%)
    const double L_t = 0.07 * L;                        // Turbulence length scale
    const double k0 = 1.5 * pow(I * 0.01, 2);           // Initial turbulent kinetic energy
    const double omega0 = sqrt(k0) / (0.09 * L_t);      // Initial specific dissipation
    const double sigma_k = 0.85;
    const double sigma_omega = 0.5;
    const double beta_star = 0.09;
    const double beta = 0.075;
    const double alpha = 5.0 / 9.0;
    double Pr_t = 0.9;

    // Turbulence fields for sodium vapor initialization
    std::vector<double> k_turb(N, k0);
    std::vector<double> omega_turb(N, omega0);
    std::vector<double> mu_t(N, 0.0);

    // Models
    const int rhie_chow_on_off = 1;                 // 0: no RC correction, 1: with RC correction
    const int SST_model_turbulence_on_off = 1;      // 0: no turbulence, 1: with turbulence

    std::vector<double> aU(N, 0.0), bU(N, 2 * D + dz / dt * rho[0]), cU(N, 0.0), dU(N, 0.0);

    #pragma endregion

    // Output file
    std::ofstream fout("solution_rhoPISO.txt");

    for (double it = 0; it < t_iter; it++) {

        std::cout << "Solving! Time elapsed:" << dt * it << "/" << t_max 
                        << ", max courant number: " << *std::max_element(u.begin(),u.end()) * dt / dz
                        << ", max reynolds number: " << *std::max_element(u.begin(), u.end()) * D_pipe * *std::max_element(rho.begin(), rho.end()) / mu << "\n";

        // Backup variables
        T_old = T;
        rho_old = rho;
        p_old = p;

        // PISO pressure correction loops
        double maxErr = 1.0;
        int iter = 0;

        while (iter<tot_iter && maxErr>tol) {

            // =======================================================================
            //
            //                      [MOMENTUM PREDICTOR]
            //
            // =======================================================================

            #pragma region momentum_predictor

            // #pragma omp parallel
            for (int i = 2; i < N - 2; i++) {

                double rhie_chow_left = - (1.0 / bU[i - 1] + 1.0 / bU[i]) / (8 * dz) * (p[i - 2] - 3 * p[i - 1] + 3 * p[i] - p[i + 1]);
                double rhie_chow_right = - (1.0 / bU[i + 1] + 1.0 / bU[i]) / (8 * dz) * (p[i - 1] - 3 * p[i] + 3 * p[i + 1] - p[i + 2]);

                double u_left_face = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_left;
                double u_right_face = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_right;

                if (u_left_face >= 0 && u_right_face >= 0) {

                    aU[i] = -u_left_face * rho[i - 1] - D;
                    cU[i] = -D;
                    bU[i] = u_right_face * rho[i] + rho[i] * dz / dt + 2 * D;
                    dU[i] = -0.5 * (p[i + 1] - p[i - 1]) + rho[i] * u[i] * dz / dt + Su[i] * dz;

                }
                else if (u_left_face >= 0 && u_right_face < 0) {

                    aU[i] = -u_left_face * rho[i - 1] - D;
                    cU[i] = u_right_face * rho[i + 1] - D;
                    bU[i] = rho[i] * dz / dt + 2 * D;
                    dU[i] = -0.5 * (p[i + 1] - p[i - 1]) + rho[i] * u[i] * dz / dt + Su[i] * dz;

                }
                else if (u_left_face < 0 && u_right_face >= 0) {

                    aU[i] = -D;
                    cU[i] = -D;
                    bU[i] = (u_right_face - u_left_face) * rho[i] + rho[i] * dz / dt + 2 * D;
                    dU[i] = -0.5 * (p[i + 1] - p[i - 1]) + rho[i] * u[i] * dz / dt + Su[i] * dz;

                }
                else if (u_left_face < 0 && u_right_face < 0) {

                    aU[i] = -D;
                    cU[i] = u_right_face * rho[i + 1] - D;
                    bU[i] = -u_left_face * rho[i] + rho[i] * dz / dt + 2 * D;
                    dU[i] = -0.5 * (p[i + 1] - p[i - 1]) + rho[i] * u[i] * dz / dt + Su[i] * dz;

                }

                printf("");
            }

            // Velocity BC: Dirichlet at left, dirichlet at right
            bU[0] = rho[0] * dz / dt + 2 * D; cU[0] = 0.0; dU[0] = (rho[0] * dz / dt + 2 * D) * u_inlet;
            aU[N - 1] = 0.0; bU[N - 1] = rho[N - 1] * dz / dt + 2 * D; dU[N - 1] = (rho[N - 1] * dz / dt + 2 * D) * u_outlet;

            // Second node
            double rhie_chow_left_second = -(1.0 / bU[0] + 1.0 / bU[1]) / (8 * dz) * (-2 * p[0] + 3 * p[1] - p[2]);
            double rhie_chow_right_second = -(1.0 / bU[2] + 1.0 / bU[1]) / (8 * dz) * (p[0] - 3 * p[1] + 3 * p[2] - p[3]);

            double u_left_face_second = 0.5 * (u[0] + u[1]) + rhie_chow_on_off * rhie_chow_left_second;
            double u_right_face_second = 0.5 * (u[1] + u[2]) + rhie_chow_on_off * rhie_chow_right_second;

            if (u_left_face_second >= 0 && u_right_face_second >= 0) {

                aU[1] = -u_left_face_second * rho[0] - D;
                cU[1] = -D;
                bU[1] = u_right_face_second * rho[1] + rho[1] * dz / dt + 2 * D;
                dU[1] = -0.5 * (p[2] - p[0]) + rho[1] * u[1] * dz / dt + Su[1] * dz;

            }
            else if (u_left_face_second >= 0 && u_right_face_second < 0) {

                aU[1] = -u_left_face_second * rho[0] - D;
                cU[1] = u_right_face_second * rho[2] - D;
                bU[1] = rho[1] * dz / dt + 2 * D;
                dU[1] = -0.5 * (p[2] - p[0]) + rho[1] * u[1] * dz / dt + Su[1] * dz;

            }
            else if (u_left_face_second < 0 && u_right_face_second >= 0) {

                aU[1] = -D;
                cU[1] = -D;
                bU[1] = (u_right_face_second - u_left_face_second) * rho[1] + rho[1] * dz / dt + 2 * D;
                dU[1] = -0.5 * (p[2] - p[0]) + rho[1] * u[1] * dz / dt + Su[1] * dz;

            }
            else if (u_left_face_second < 0 && u_right_face_second < 0) {

                aU[1] = -D;
                cU[1] = u_right_face_second * rho[2] - D;
                bU[1] = -u_left_face_second * rho[1] + rho[1] * dz / dt + 2 * D;
                dU[1] = -0.5 * (p[2] - p[0]) + rho[1] * u[1] * dz / dt + Su[1] * dz;

            }

            // Second-to-last node
            double rhie_chow_left_second_to_last = -(1.0 / bU[N - 3] + 1.0 / bU[N - 2]) / (8 * dz) * (p[N - 4] - 3 * p[N - 3] + 2 * p_outlet);
            double rhie_chow_right_second_to_last = -(1.0 / bU[N - 1] + 1.0 / bU[N - 2]) / (8 * dz) * (p[N - 3] - 3 * p[N - 2] + 2 * p_outlet);

            double u_left_face_second_to_last = 0.5 * (u[N - 3] + u[N - 2]) + rhie_chow_on_off * rhie_chow_left_second_to_last;
            double u_right_face_second_to_last = 0.5 * (u[N - 2] + u[N - 1]) + rhie_chow_on_off * rhie_chow_right_second_to_last;

            if (u_left_face_second_to_last >= 0 && u_right_face_second_to_last >= 0) {

                aU[N - 2] = -u_left_face_second_to_last * rho[N - 3] - D;
                cU[N - 2] = -D;
                bU[N - 2] = u_right_face_second_to_last * rho[N - 2] + rho[N - 2] * dz / dt + 2 * D;
                dU[N - 2] = -0.5 * (p[N - 1] - p[N - 3]) + rho[N - 2] * u[N - 2] * dz / dt + Su[N - 2] * dz;

            }
            else if (u_left_face_second_to_last >= 0 && u_right_face_second_to_last < 0) {

                aU[N - 2] = -u_left_face_second_to_last * rho[N - 3] - D;
                cU[N - 2] = u_right_face_second_to_last * rho[N - 1] - D;
                bU[N - 2] = rho[N - 2] * dz / dt + 2 * D;
                dU[N - 2] = -0.5 * (p[N - 1] - p[N - 3]) + rho[N - 2] * u[N - 2] * dz / dt + Su[N - 2] * dz;

            }
            else if (u_left_face_second_to_last < 0 && u_right_face_second_to_last >= 0) {

                aU[N - 2] = -D;
                cU[N - 2] = -D;
                bU[N - 2] = (u_right_face_second_to_last - u_left_face_second_to_last) * rho[N - 2] + rho[N - 2] * dz / dt + 2 * D;
                dU[N - 2] = -0.5 * (p[N - 1] - p[N - 3]) + rho[N - 2] * u[N - 2] * dz / dt + Su[N - 2] * dz;

            }
            else if (u_left_face_second_to_last < 0 && u_right_face_second_to_last < 0) {

                aU[N - 2] = -D;
                cU[N - 2] = u_right_face_second_to_last * rho[N - 1] - D;
                bU[N - 2] = -u_left_face_second_to_last * rho[N - 2] + rho[N - 2] * dz / dt + 2 * D;
                dU[N - 2] = -0.5 * (p[N - 1] - p[N - 3]) + rho[N - 2] * u[N - 2] * dz / dt + Su[N - 2] * dz;

            }

            #pragma endregion

            u = solveTridiagonal(aU, bU, cU, dU);

            for (int piso = 0; piso < corr_iter; piso++) {

                // =======================================================================
                //
                //                       [PRESSURE CORRECTOR]
                //
                // =======================================================================

                #pragma region pressure_corrector

                std::vector<double> aP(N, 0.0), bP(N, 0.0), cP(N, 0.0), dP(N, 0.0);

                // // #pragma omp parallel
                for (int i = 2; i < N - 2; i++) {

                    double rhie_chow_left = -(1.0 / bU[i - 1] + 1.0 / bU[i]) / (8 * dz) * (p[i - 2] - 3 * p[i - 1] + 3 * p[i] - p[i + 1]);
                    double rhie_chow_right = -(1.0 / bU[i + 1] + 1.0 / bU[i]) / (8 * dz) * (p[i - 1] - 3 * p[i] + 3 * p[i + 1] - p[i + 2]);

                    double rho_w = 0.5 * (rho[i - 1] + rho[i]);
                    double d_w_face = 0.5 * (1.0 / bU[i - 1] + 1.0 / bU[i]); // 1/Ap average on west face
                    double E_w = rho_w * d_w_face / (dz * dz);

                    double rho_e = 0.5 * (rho[i] + rho[i + 1]);
                    double d_e_face = 0.5 * (1.0 / bU[i] + 1.0 / bU[i + 1]);  // 1/Ap average on east face
                    double E_e = rho_e * d_e_face / (dz * dz);

                    double psi_i = 1.0 / (Rv * T[i]); // Compressibility assuming ideal gas

                    double u_w_star = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_left;
                    double mdot_w_star = (u_w_star > 0.0) ? rho[i - 1] * u_w_star : rho[i] * u_w_star;

                    double u_e_star = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_right;
                    double mdot_e_star = (u_e_star > 0.0) ? rho[i] * u_e_star : rho[i + 1] * u_e_star;

                    double mass_imbalance = (rho[i] - rho_old[i]) / dt + (mdot_e_star - mdot_w_star) / dz;

                    aP[i] = -E_w;
                    cP[i] = -E_e;
                    bP[i] = E_w + E_e + psi_i / dt;
                    dP[i] = Sm[i] - mass_imbalance;

                    printf("");
                }

                // BCs for p': zero gradient at inlet and zero correction at outlet
                bP[0] = 1.0; cP[0] = -1.0; dP[0] = 0.0;
                bP[N - 1] = 1.0; aP[N - 1] = 0.0; dP[N - 1] = 0.0;

                // Second node
                double rhie_chow_left_second = -(1.0 / bU[0] + 1.0 / bU[1]) / (8 * dz) * (- 2 * p[0] + 3 * p[1] - p[2]);
                double rhie_chow_right_second = -(1.0 / bU[2] + 1.0 / bU[1]) / (8 * dz) * (p[0] - 3 * p[1] + 3 * p[2] - p[3]);

                double rho_w_second = 0.5 * (rho[0] + rho[1]);
                double d_w_face_second = 0.5 * (1.0 / bU[0] + 1.0 / bU[1]); // 1/Ap average on west face
                double E_w_second = rho_w_second * d_w_face_second / (dz * dz);

                double rho_e_second = 0.5 * (rho[1] + rho[2]);
                double d_e_face_second = 0.5 * (1.0 / bU[1] + 1.0 / bU[2]);  // 1/Ap average on east face
                double E_e_second = rho_e_second * d_e_face_second / (dz * dz);

                double psi_i_second = 1.0 / (Rv * T[1]); // Compressibility assuming ideal gas

                double u_w_star_second = 0.5 * (u[0] + u[1]) + rhie_chow_on_off * rhie_chow_left_second;
                double mdot_w_star_second = (u_w_star_second > 0.0) ? rho[0] * u_w_star_second : rho[1] * u_w_star_second;

                double u_e_star_second = 0.5 * (u[1] + u[2]) + rhie_chow_on_off * rhie_chow_right_second;
                double mdot_e_star_second = (u_e_star_second > 0.0) ? rho[1] * u_e_star_second : rho[2] * u_e_star_second;

                double mass_imbalance_second = (rho[1] - rho_old[1]) / dt + (mdot_e_star_second - mdot_w_star_second) / dz;

                aP[1] = -E_w_second;
                cP[1] = -E_e_second;
                bP[1] = E_w_second + E_e_second + psi_i_second / dt;
                dP[1] = Sm[1] - mass_imbalance_second;

                // Second-to-last node
                double rhie_chow_left_second_to_last = -(1.0 / bU[N - 3] + 1.0 / bU[N - 2]) / (8 * dz) * (p[N - 4] - 3 * p[N - 3] + 3 * p[N - 2] - p[N - 1]);
                double rhie_chow_right_second_to_last = -(1.0 / bU[N - 1] + 1.0 / bU[N - 2]) / (8 * dz) * (p[N - 3] - 3 * p[N - 2] + 2 * p[N - 1]);

                double rho_w_second_to_last = 0.5 * (rho[N - 3] + rho[N - 2]);
                double d_w_face_second_to_last = 0.5 * (1.0 / bU[N - 3] + 1.0 / bU[N - 2]); // 1/Ap average on west face
                double E_w_second_to_last = rho_w_second_to_last * d_w_face_second_to_last / (dz * dz);

                double rho_e_second_to_last = 0.5 * (rho[N - 2] + rho[N - 1]);
                double d_e_face_second_to_last = 0.5 * (1.0 / bU[N - 2] + 1.0 / bU[N - 1]);  // 1/Ap average on east face
                double E_e_second_to_last = rho_e_second_to_last * d_e_face_second_to_last / (dz * dz);

                double psi_i_second_to_last = 1.0 / (Rv * T[N - 2]); // Compressibility assuming ideal gas

                double u_w_star_second_to_last = 0.5 * (u[N - 3] + u[N - 2]) + rhie_chow_on_off * rhie_chow_left_second_to_last;
                double mdot_w_star_second_to_last = (u_w_star_second_to_last > 0.0) ? rho[N - 3] * u_w_star_second_to_last : rho[N - 2] * u_w_star_second_to_last;

                double u_e_star_second_to_last = 0.5 * (u[N - 2] + u[N - 1]) + rhie_chow_on_off * rhie_chow_right_second_to_last;
                double mdot_e_star_second_to_last = (u_e_star_second_to_last > 0.0) ? rho[N - 2] * u_e_star_second_to_last : rho[N - 1] * u_e_star_second_to_last;

                double mass_imbalance_second_to_last = (rho[N - 2] - rho_old[N - 2]) / dt + (mdot_e_star_second_to_last - mdot_w_star_second_to_last) / dz;

                aP[N - 2] = -E_w_second_to_last;
                cP[N - 2] = -E_e_second_to_last;
                bP[N - 2] = E_w_second_to_last + E_e_second_to_last + psi_i_second_to_last / dt;
                dP[N - 2] = Sm[N - 2] - mass_imbalance_second_to_last;

                #pragma endregion

                p_prime = solveTridiagonal(aP, bP, cP, dP);

                // =======================================================================
                //
                //                        [PRESSURE UPDATER]
                //
                // =======================================================================

                for (int i = 0; i < N; i++) p[i] += p_prime[i]; // Note that PISO does not require an under-relaxation factor

                // =======================================================================
                //
                //                        [VELOCITY UPDATER]
                //
                // =======================================================================

                #pragma region velocity_updater

                maxErr = 0.0;
                for (int i = 1; i < N - 1; i++) {

                    double u_prev = u[i];
                    u[i] = u[i] - (p_prime[i + 1] - p_prime[i - 1]) / (2.0 * dz * bU[i]);
                    maxErr = std::max(maxErr, std::fabs(u[i] - u_prev));
                }

                #pragma endregion

                // Enforcing velocity BCs
                u[0] = 0.0;
                u[N - 1] = 0.0;

                // Enforcing pressure BCs
                p[0] = p[1];
                p[N - 1] = p_outlet; 

            }

            iter++;
        }

        // Update density with new p,T
        eos_update(rho, p, T);

        // =======================================================================
        //
        //                        [TURBULENCE MODEL]
        //
        // =======================================================================

        #pragma region turbulence_SST

        if(SST_model_turbulence_on_off == 1) {

            // --- Turbulence transport equations (1D implicit form) ---
            const double sigma_k = 0.85;
            const double sigma_omega = 0.5;
            const double beta_star = 0.09;
            const double beta = 0.075;
            const double alpha = 5.0 / 9.0;

            std::vector<double> aK(N, 0.0), bK(N, 0.0), cK(N, 0.0), dK(N, 0.0);
            std::vector<double> aW(N, 0.0), bW(N, 0.0), cW(N, 0.0), dW(N, 0.0);

            // --- Compute strain rate and production ---
            std::vector<double> dudz(N, 0.0);
            std::vector<double> Pk(N, 0.0);

            for (int i = 1; i < N - 1; i++) {
                dudz[i] = (u[i + 1] - u[i - 1]) / (2.0 * dz);
                Pk[i] = mu_t[i] * pow(dudz[i], 2.0);
            }

            // --- k-equation ---
            for (int i = 1; i < N - 1; i++) {
                double mu_eff = mu + mu_t[i];
                double Dw = mu_eff / (sigma_k * dz * dz);
                double De = mu_eff / (sigma_k * dz * dz);
                       aK[i] = -Dw;
                cK[i] = -De;
                bK[i] = rho[i] / dt + Dw + De + beta_star * rho[i] * omega_turb[i];
                dK[i] = rho[i] / dt * k_turb[i] + Pk[i];
            }
        
            // k BCs: constant initial values at the boundaries
            bK[0] = 1.0; dK[0] = k_turb[0]; cK[0] = 0.0;
            aK[N - 1] = 0.0; bK[N - 1] = 1.0; dK[N - 1] = k_turb[N - 1];

            k_turb = solveTridiagonal(aK, bK, cK, dK);

            // --- omega-equation ---
            for (int i = 1; i < N - 1; i++) {
                double mu_eff = mu + mu_t[i];
                double Dw = mu_eff / (sigma_omega * dz * dz);
                double De = mu_eff / (sigma_omega * dz * dz);

                aW[i] = -Dw;
                cW[i] = -De;
                bW[i] = rho[i] / dt + Dw + De + beta * rho[i] * omega_turb[i];
                dW[i] = rho[i] / dt * omega_turb[i] + alpha * (omega_turb[i] / k_turb[i]) * Pk[i];
            }
            bW[0] = 1.0; dW[0] = omega_turb[0]; cW[0] = 0.0;
            aW[N - 1] = 0.0; bW[N - 1] = 1.0; dW[N - 1] = omega_turb[N - 1];

            omega_turb = solveTridiagonal(aW, bW, cW, dW);

            // --- Update turbulent viscosity ---
            for (int i = 0; i < N; i++) {
                double denom = std::max(omega_turb[i], 1e-6);
                mu_t[i] = rho[i] * k_turb[i] / denom;
                mu_t[i] = std::min(mu_t[i], 1000.0 * mu); // limiter
            }
        }

        #pragma endregion

        // =======================================================================
        //
        //                        [TEMPERATURE CALCULATOR]
        //
        // =======================================================================

        #pragma region temperature_calculator

        // Energy equation for T (implicit), upwind convection, central diffusion
        std::vector<double> aT(N, 0.0), bT(N, 0.0), cT(N, 0.0), dT(N, 0.0);

        // #pragma omp parallel
        for (int i = 2; i < N - 2; i++) {

            double rhoCp_dt = rho_old[i] * cp / dt; // Termine transiente
            double keff = k_cond + SST_model_turbulence_on_off * (mu_t[i] * cp / Pr_t);

            double D_w = keff / (dz * dz); // Unità: W/(m^3 K)
            double D_e = keff / (dz * dz); // Unità: W/(m^3 K)

            double rhie_chow_left = -(1.0 / bU[i - 1] + 1.0 / bU[i]) / (8 * dz) * (p[i - 2] - 3 * p[i - 1] + 3 * p[i] - p[i + 1]);
            double rhie_chow_right = -(1.0 / bU[i + 1] + 1.0 / bU[i]) / (8 * dz) * (p[i - 1] - 3 * p[i] + 3 * p[i + 1] - p[i + 2]);

            double u_left_face = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_left;
            double u_right_face = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_right;

            double rho_w = (u_left_face >= 0) ? rho[i - 1] : rho[i];
            double rho_e = (u_right_face >= 0) ? rho[i] : rho[i + 1];

            double Fw = rho_w * u_left_face;
            double Fe = rho_e * u_right_face;

            double C_w_dx = (Fw * cp) / dz;
            double C_e_dx = (Fe * cp) / dz;

            double A_w = D_w + std::max(C_w_dx, 0.0);
            double A_e = D_e + std::max(-C_e_dx, 0.0);

            aT[i] = - A_w;
            cT[i] = - A_e;
            bT[i] = A_w + A_e + rhoCp_dt;

            double pressure_work = (p[i] - p_old[i]) / dt;
            dT[i] = rhoCp_dt * T_old[i] + pressure_work;

        }

        double rhoCp_dt = rho_old[1] * cp / dt; // Termine transiente
        double keff = k_cond + SST_model_turbulence_on_off * (mu_t[N - 2] * cp / Pr_t);

        // Second node
        double D_w_second = keff / (dz * dz); // Unità: W/(m^3 K)
        double D_e_second = keff / (dz * dz); // Unità: W/(m^3 K)

        double rhie_chow_left_second = -(1.0 / bU[0] + 1.0 / bU[1]) / (8 * dz) * (-2 * p[0] + 3 * p[1] - p[2]);
        double rhie_chow_right_second = -(1.0 / bU[2] + 1.0 / bU[1]) / (8 * dz) * (p[0] - 3 * p[1] + 3 * p[2] - p[3]);

        double u_left_face_second = 0.5 * (u[0] + u[1]) + rhie_chow_on_off * rhie_chow_left_second;
        double u_right_face_second = 0.5 * (u[1] + u[2]) + rhie_chow_on_off * rhie_chow_right_second;

        double rho_w_second = (u_left_face_second >= 0) ? rho[0] : rho[1];
        double rho_e_second = (u_right_face_second >= 0) ? rho[1] : rho[2];

        double Fw_second = rho_w_second * u_left_face_second;
        double Fe_second = rho_e_second * u_right_face_second;

        double C_w_dx_second = (Fw_second * cp) / dz;
        double C_e_dx_second = (Fe_second * cp) / dz;

        double A_w_second = D_w_second + std::max(C_w_dx_second, 0.0);
        double A_e_second = D_e_second + std::max(-C_e_dx_second, 0.0);

        aT[1] = -A_w_second;
        cT[1] = -A_e_second;
        bT[1] = A_w_second + A_e_second + rhoCp_dt;

        double pressure_work_second = (p[1] - p_old[1]) / dt;
        dT[1] = rhoCp_dt * T_old[1] + pressure_work_second;

        // Second-to-last node
        double rhoCp_dt_second_to_last = rho_old[N - 2] * cp / dt; // Termine transiente
        double keff_second_to_last = k_cond + SST_model_turbulence_on_off * (mu_t[N - 2] * cp / Pr_t);

        double D_w_second_to_last = keff / (dz * dz); // Unità: W/(m^3 K)
        double D_e_second_to_last = keff / (dz * dz); // Unità: W/(m^3 K)

        double rhie_chow_left_second_to_last = -(1.0 / bU[N - 3] + 1.0 / bU[N - 2]) / (8 * dz) * (p[N - 4] - 3 * p[N - 3] + 3 * p[N - 2] - p[N - 1]);
        double rhie_chow_right_second_to_last = -(1.0 / bU[N - 1] + 1.0 / bU[N - 2]) / (8 * dz) * (p[N - 3] - 3 * p[N - 2] + 2 * p[N - 1]);

        double u_left_face_second_to_last = 0.5 * (u[N - 3] + u[N - 2]) + rhie_chow_on_off * rhie_chow_left_second_to_last;
        double u_right_face_second_to_last = 0.5 * (u[N - 2] + u[N - 1]) + rhie_chow_on_off * rhie_chow_right_second_to_last;

        double rho_w_second_to_last = (u_left_face_second_to_last >= 0) ? rho[N - 3] : rho[N - 2];
        double rho_e_second_to_last = (u_right_face_second_to_last >= 0) ? rho[N - 2] : rho[N - 1];

        double Fw_second_to_last = rho_w_second_to_last * u_left_face_second_to_last;
        double Fe_second_to_last = rho_e_second_to_last * u_right_face_second_to_last;

        double C_w_dx_second_to_last = (Fw_second_to_last * cp) / dz;
        double C_e_dx_second_to_last = (Fe_second_to_last * cp) / dz;

        double A_w_second_to_last = D_w_second_to_last + std::max(C_w_dx_second_to_last, 0.0);
        double A_e_second_to_last = D_e_second_to_last + std::max(-C_e_dx_second_to_last, 0.0);

        aT[N - 2] = -A_w_second_to_last;
        cT[N - 2] = -A_e_second_to_last;
        bT[N - 2] = A_w_second_to_last + A_e_second_to_last + rhoCp_dt;

        double pressure_work_second_to_last = (p[N - 2] - p_old[N - 2]) / dt;
        dT[N - 2] = rhoCp_dt * T_old[N - 2] + pressure_work_second_to_last;

        // Temperature BCs
        bT[0] = 1.0; cT[0] = 0.0; dT[0] = T_inlet;
        aT[N - 1] = 0.0; bT[N - 1] = 1.0; dT[N - 1] = T_outlet;

        T = solveTridiagonal(aT, bT, cT, dT);

        #pragma endregion

        // Update density with new p,T
        eos_update(rho, p, T);

        // =======================================================================
        //
        //                                [OUTPUT]
        //
        // =======================================================================

        #pragma region output

        // Write last step profiles
        if (it == (t_iter - 1)) {
            for (int i = 0; i < N; i++) {
                fout << u[i] << ", ";
            }
        } fout << "\n\n";

        // Write last step profiles
        if (it == (t_iter - 1)) {
            for (int i = 0; i < N; i++) {
                fout << p[i] << ", ";
            }
        } fout << "\n\n";

        // Write last step profiles
        if (it == (t_iter - 1)) {
            for (int i = 0; i < N; i++) {
                fout << T[i] << ", ";
            }
        }

        #pragma endregion
    }

    fout.close();

    return 0;
}
