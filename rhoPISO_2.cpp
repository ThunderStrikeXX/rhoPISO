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
    const int    N = 100;                      // Number of nodes [-]
    const double dz = L / (N - 1);             // Grid spacing [m]

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
    std::vector<double> T_old(N, 380.0), rho_old(N, 0.5);
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
    const double I = 0.05;                          // Turbulence intensity (5%)
    const double L_t = 0.07 * L;                    // Turbulence length scale
    const double k0 = 1.5 * pow(I * 0.01, 2);        // Initial turbulent kinetic energy
    const double omega0 = sqrt(k0) / (0.09 * L_t);        // Initial specific dissipation
    const double sigma_k = 0.85;
    const double sigma_omega = 0.5;
    const double beta_star = 0.09;
    const double beta = 0.075;
    const double alpha = 5.0 / 9.0;

    // Turbulence fields for sodium vapor initialization
    std::vector<double> k_turb(N, k0);
    std::vector<double> omega_turb(N, omega0);
    std::vector<double> mu_t(N, 0.0);

    // Models
    const int rhie_chow_on_off = 0;                 // 0: no RC correction, 1: with RC correction
    const int SST_model_turbulence_on_off = 0;      // 0: no turbulence, 1: with turbulence

    #pragma endregion

    // Output file
    std::ofstream fout("solution_rhoPISO.txt");

    for (double it = 0; it < t_iter; it++) {

        std::cout << "Solving! Time elapsed:" << dt * it << "/" << t_max << ", max courant number: " << *std::max_element(u.begin(),u.end()) * dt / dz << "\n";

        // Backup variables
        T_old = T;
        rho_old = rho;

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
            std::vector<double> aU(N, 0.0), bU(N, 2 * 4.0 / 3.0 * mu / dz + dz / dt * rho[0]), cU(N, 0.0), dU(N, 0.0);

            // #pragma omp parallel
            for (int i = 2; i < N - 2; i++) {

                const double D = 4.0 / 3.0 * mu / dz;

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

            // Velocity BC: Dirichlet at left, zero-gradient at right
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
                for (int i = 1; i < N - 1; i++) {

                    // --- Coefficienti di diffusione di pressione ---

                    // Faccia Ovest (w, tra i-1 e i)
                    double rho_w = 0.5 * (rho[i - 1] + rho[i]);
                    // 1/Ap medio sulla faccia
                    double d_w_face = 0.5 * (1.0 / bU[i - 1] + 1.0 / bU[i]);
                    // Coeff. ellittico (nota: dz*dz)
                    double E_w = rho_w * d_w_face / (dz * dz);

                    // Faccia Est (e, tra i e i+1)
                    double rho_e = 0.5 * (rho[i] + rho[i + 1]);
                    // 1/Ap medio sulla faccia
                    double d_e_face = 0.5 * (1.0 / bU[i] + 1.0 / bU[i + 1]);
                    // Coeff. ellittico
                    double E_e = rho_e * d_e_face / (dz * dz);

                    // --- Compressibilità (psi = d(rho)/d(p)) ---
                    // Assumendo gas ideale: rho = p/(RT) -> d(rho)/d(p) = 1/(RT)
                    double psi_i = 1.0 / (Rv * T[i]); // T è T* (dal passo precedente)

                    // --- Calcolo sbilancio di massa (RHS) ---
                    // (Usa una semplice media per u* e upwind per rho*)
                    double u_w_star = 0.5 * (u[i - 1] + u[i]);
                    double mdot_w_star = (u_w_star > 0.0) ? rho[i - 1] * u_w_star : rho[i] * u_w_star;

                    double u_e_star = 0.5 * (u[i] + u[i + 1]);
                    double mdot_e_star = (u_e_star > 0.0) ? rho[i] * u_e_star : rho[i + 1] * u_e_star;

                    double mass_imbalance = (rho[i] - rho_old[i]) / dt + (mdot_e_star - mdot_w_star) / dz;

                    // --- Assembla Matrice P_PRIME ---
                    aP[i] = -E_w;
                    cP[i] = -E_e;
                    bP[i] = E_w + E_e + psi_i / dt;
                    dP[i] = Sm[i] - mass_imbalance; // S_m - (sbilancio di massa)

                    //double rhie_chow_left = -(1.0 / bU[i - 1] + 1.0 / bU[i]) / (8 * dz) * (p[i - 2] - 3 * p[i - 1] + 3 * p[i] - p[i + 1]);
                    //double rhie_chow_right = -(1.0 / bU[i + 1] + 1.0 / bU[i]) / (8 * dz) * (p[i - 1] - 3 * p[i] + 3 * p[i + 1] - p[i + 2]);

                    //double u_left_face = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_left;
                    //double u_right_face = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_right;

                    //double rhoW = (u_left_face >= 0) ? rho[i - 1] : rho[i];
                    //double rhoE = (u_right_face >= 0) ? rho[i] : rho[i + 1];

                    //double D_W = 0.5 * rhoW * (1.0 / bU[i - 1] + 1.0 / bU[i]) / dz;
                    //double D_E = 0.5 * rhoE * (1.0 / bU[i + 1] + 1.0 / bU[i]) / dz;

                    //double F_W = rhoW * u_left_face;
                    //double F_E = rhoE * u_right_face;

                    //aP[i] = -D_W;
                    //cP[i] = -D_E;
                    //bP[i] = D_E + D_W + rho[i] / p[i] * dz / dt;
                    //dP[i] = F_E - F_W + Sm[i] * dz;

                    printf("");
                }

                // BCs for p'
                // Left: zero gradient correction
                bP[0] = 1.0; cP[0] = -1.0; dP[0] = 0.0;

                // Right: p' = 0 to pin pressure level
                bP[N - 1] = 1.0; aP[N - 1] = 0.0; dP[N - 1] = 0.0;

                //// a_P coefficients for first and last node
                //double bU0 = rho[0] * dz / dt + 2 * D;
                //double bU_N_1 = rho[N - 1] * dz / dt + 2 * D;

                //// Second node
                //double rhie_chow_left_second = -(1.0 / bU0 + 1.0 / bU[1]) / (8 * dz) * (- 2 * p[0] + 3 * p[1] - p[2]);
                //double rhie_chow_right_second = -(1.0 / bU[2] + 1.0 / bU[1]) / (8 * dz) * (p[0] - 3 * p[1] + 3 * p[2] - p[3]);

                //double u_left_face = 0.5 * (u[0] + u[1]) + rhie_chow_on_off * rhie_chow_left_second;
                //double u_right_face = 0.5 * (u[1] + u[2]) + rhie_chow_on_off * rhie_chow_right_second;

                //double rhoW_second = (u_left_face_second >= 0) ? rho[0] : rho[1];
                //double rhoE_second = (u_right_face_second >= 0) ? rho[1] : rho[2];

                //double D_W_second = 0.5 * rhoW_second * (1.0 / bU0 + 1.0 / bU[1]) / dz;
                //double D_E_second = 0.5 * rhoE_second * (1.0 / bU[2] + 1.0 / bU[1]) / dz;

                //double F_W_second = rhoW_second * u_left_face_second;
                //double F_E_second = rhoE_second * u_right_face_second;

                //aP[1] = -D_W_second;
                //cP[1] = -D_E_second;
                //bP[1] = D_E_second + D_W_second + rho[1] / p[1] * dz / dt;
                //dP[1] = F_E_second - F_W_second + Sm[1] * dz;

                //// Second-to-last node
                //double rhie_chow_left_second_to_last = -(1.0 / bU[N - 3] + 1.0 / bU[N - 2]) / (8 * dz) * (p[N - 4] - 3 * p[N - 3] + 3 * p[N - 2] - p[N - 1]);
                //double rhie_chow_right_second_to_last = -(1.0 / bU_N_1 + 1.0 / bU[N - 2]) / (8 * dz) * (p[N - 3] - 3 * p[N - 2] + 2 * p[N - 1]);

                //double u_left_face_second_to_last = 0.5 * (u[N - 3] + u[N - 2]) + rhie_chow_on_off * rhie_chow_left_second_to_last;
                //double u_right_face_second_to_last = 0.5 * (u[N - 2] + u[N - 1]) + rhie_chow_on_off * rhie_chow_right_second_to_last;

                //double rhoW_second_to_last = (u_left_face_second_to_last >= 0) ? rho[N - 3] : rho[N - 2];
                //double rhoE_second_to_last = (u_right_face_second_to_last >= 0) ? rho[N - 2] : rho[N - 1];

                //double D_W_second_to_last = 0.5 * rhoW_second_to_last * (1.0 / bU[N - 3] + 1.0 / bU[N - 2]) / dz;
                //double D_E_second_to_last = 0.5 * rhoE_second_to_last * (1.0 / bU_N_1 + 1.0 / bU[N - 2]) / dz;

                //double F_W_second_to_last = rhoW_second_to_last * u_left_face_second_to_last;
                //double F_E_second_to_last = rhoE_second_to_last * u_right_face_second_to_last;

                //aP[N - 2] = -D_W_second_to_last;
                //cP[N - 2] = -D_E_second_to_last;
                //bP[N - 2] = D_E_second_to_last + D_W_second_to_last + rho[N - 2] / p[N - 2] * dz / dt;
                //dP[N - 2] = F_E_second_to_last - F_W_second_to_last + Sm[N - 2] * dz;

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
        for (int i = 1; i < N - 1; i++) {
            double rhoCp = rho[i] * cp;
            double Pr_t = 0.9;
            double keff = k_cond + SST_model_turbulence_on_off * (mu_t[i] * cp / Pr_t);
            double Dw = keff / (dz * dz);
            double De = keff / (dz * dz);

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
