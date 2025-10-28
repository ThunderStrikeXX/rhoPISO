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
    std::vector<double> u(Nx, 0.01), p(Nx, 50000.0), T(Nx, 380.0), rho(Nx, 0.5);
    std::vector<double> T_old(Nx, 380.0), rho_old(Nx, 0.5);
    std::vector<double> p_prime(Nx, 0.0);

    // Boundary conditions (Dirichlet p at outlet, T at both ends, u inlet)
    const double u_inlet = 0.0;        // Inlet velocity [m/s]
    const double u_outlet = 0.0;        // Outlet velocity [m/s]
    const double p_outlet = 50000.0;  // Outlet pressure [Pa]
    const double T_inlet = 390.0;    // Inlet temperature [K] (evaporator)
    const double T_outlet = 350.0;   // Outlet temperature [K] (condenser)

    auto eos_update = [&](std::vector<double>& rho_, const std::vector<double>& p_, const std::vector<double>& T_) {
    // #pragma omp parallel
        for (int i = 0; i < Nx; i++) {
            double Ti = std::max(200.0, T_[i]);
            rho_[i] = std::max(1e-6, p_[i] / (Rv * Ti));
        }
    }; eos_update(rho, p, T);

    // Mass source
    std::vector<double> Sm(Nx, 0.0);
    Sm[1] = 10.0; // mass source term at cell 1
    Sm[2] = 10.0; // mass source term at cell 1
    Sm[7] = -10.0; // mass sink term at cell 8
    Sm[8] = -10.0; // mass sink term at cell 8

    // Momentum source
    std::vector<double> Su(Nx, 0.0);

    const double D = 4.0 / 3.0 * mu / dx;

    const double rhie_chow_on_off = 0;

    // Output
    std::ofstream fout("solution_vapor_PISO_thermal.txt");

    for (double it = 0; it < t_iter; it++) {

        T_old = T;
        rho_old = rho;

        printf("Time: %f / %f, Courant number: %f \n", dt * it, t_max, u_inlet * dt / dx);

        // PISO pressure correction loops
        double maxErr = 1.0;
        int iter = 0;

        while (iter<tot_iter && maxErr>tol) {

            // Momentum predictort
            std::vector<double> aU(Nx, 0.0), bU(Nx, 2 * 4.0 / 3.0 * mu / dx + dx / dt * rho[0]), cU(Nx, 0.0), dU(Nx, 0.0);

            // #pragma omp parallel
            for (int i = 2; i < Nx - 2; i++) {

                const double D = 4.0 / 3.0 * mu / dx;

                double rhie_chow_left = - (1.0 / bU[i - 1] + 1.0 / bU[i]) / (8 * dx) * (p[i - 2] - 3 * p[i - 1] + 3 * p[i] - p[i + 1]);
                double rhie_chow_right = - (1.0 / bU[i + 1] + 1.0 / bU[i]) / (8 * dx) * (p[i - 1] - 3 * p[i] + 3 * p[i + 1] - p[i + 2]);

                double u_left_face = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_left;
                double u_right_face = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_right;

                if (u_left_face >= 0 && u_right_face >= 0) {

                    aU[i] = -u_left_face * rho[i - 1] - D;
                    cU[i] = -D;
                    bU[i] = u_right_face * rho[i] + rho[i] * dx / dt + 2 * D;
                    dU[i] = -0.5 * (p[i + 1] - p[i - 1]) + rho[i] * u[i] * dx / dt + Su[i] * dx;

                }
                else if (u_left_face >= 0 && u_right_face < 0) {

                    aU[i] = -u_left_face * rho[i - 1] - D;
                    cU[i] = u_right_face * rho[i + 1] - D;
                    bU[i] = rho[i] * dx / dt + 2 * D;
                    dU[i] = -0.5 * (p[i + 1] - p[i - 1]) + rho[i] * u[i] * dx / dt + Su[i] * dx;

                }
                else if (u_left_face < 0 && u_right_face >= 0) {

                    aU[i] = -D;
                    cU[i] = -D;
                    bU[i] = (u_right_face - u_left_face) * rho[i] + rho[i] * dx / dt + 2 * D;
                    dU[i] = -0.5 * (p[i + 1] - p[i - 1]) + rho[i] * u[i] * dx / dt + Su[i] * dx;

                }
                else if (u_left_face < 0 && u_right_face < 0) {

                    aU[i] = -D;
                    cU[i] = u_right_face * rho[i + 1] - D;
                    bU[i] = -u_left_face * rho[i] + rho[i] * dx / dt + 2 * D;
                    dU[i] = -0.5 * (p[i + 1] - p[i - 1]) + rho[i] * u[i] * dx / dt + Su[i] * dx;

                }

                printf("");
            }

            // Velocity BC: Dirichlet at left, zero-gradient at right
            bU[0] = rho[0] * dx / dt + 2 * D; cU[0] = 0.0; dU[0] = (rho[0] * dx / dt + 2 * D) * u_inlet;
            aU[Nx - 1] = 0.0; bU[Nx - 1] = rho[Nx - 1] * dx / dt + 2 * D; dU[Nx - 1] = (rho[Nx - 1] * dx / dt + 2 * D) * u_outlet;

            // Second node
            double rhie_chow_left_second = -(1.0 / bU[0] + 1.0 / bU[1]) / (8 * dx) * (-2 * p[0] + 3 * p[1] - p[2]);
            double rhie_chow_right_second = -(1.0 / bU[2] + 1.0 / bU[1]) / (8 * dx) * (p[0] - 3 * p[1] + 3 * p[2] - p[3]);

            double u_left_face_second = 0.5 * (u[0] + u[1]) + rhie_chow_on_off * rhie_chow_left_second;
            double u_right_face_second = 0.5 * (u[1] + u[2]) + rhie_chow_on_off * rhie_chow_right_second;

            if (u_left_face_second >= 0 && u_right_face_second >= 0) {

                aU[1] = -u_left_face_second * rho[0] - D;
                cU[1] = -D;
                bU[1] = u_right_face_second * rho[1] + rho[1] * dx / dt + 2 * D;
                dU[1] = -0.5 * (p[2] - p[0]) + rho[1] * u[1] * dx / dt + Su[1] * dx;

            }
            else if (u_left_face_second >= 0 && u_right_face_second < 0) {

                aU[1] = -u_left_face_second * rho[0] - D;
                cU[1] = u_right_face_second * rho[2] - D;
                bU[1] = rho[1] * dx / dt + 2 * D;
                dU[1] = -0.5 * (p[2] - p[0]) + rho[1] * u[1] * dx / dt + Su[1] * dx;

            }
            else if (u_left_face_second < 0 && u_right_face_second >= 0) {

                aU[1] = -D;
                cU[1] = -D;
                bU[1] = (u_right_face_second - u_left_face_second) * rho[1] + rho[1] * dx / dt + 2 * D;
                dU[1] = -0.5 * (p[2] - p[0]) + rho[1] * u[1] * dx / dt + Su[1] * dx;

            }
            else if (u_left_face_second < 0 && u_right_face_second < 0) {

                aU[1] = -D;
                cU[1] = u_right_face_second * rho[2] - D;
                bU[1] = -u_left_face_second * rho[1] + rho[1] * dx / dt + 2 * D;
                dU[1] = -0.5 * (p[2] - p[0]) + rho[1] * u[1] * dx / dt + Su[1] * dx;

            }

            // Second-to-last node
            double rhie_chow_left_second_to_last = -(1.0 / bU[Nx - 3] + 1.0 / bU[Nx - 2]) / (8 * dx) * (p[Nx - 4] - 3 * p[Nx - 3] + 2 * p_outlet);
            double rhie_chow_right_second_to_last = -(1.0 / bU[Nx - 1] + 1.0 / bU[Nx - 2]) / (8 * dx) * (p[Nx - 3] - 3 * p[Nx - 2] + 2 * p_outlet);

            double u_left_face_second_to_last = 0.5 * (u[Nx - 3] + u[Nx - 2]) + rhie_chow_on_off * rhie_chow_left_second_to_last;
            double u_right_face_second_to_last = 0.5 * (u[Nx - 2] + u[Nx - 1]) + rhie_chow_on_off * rhie_chow_right_second_to_last;

            if (u_left_face_second_to_last >= 0 && u_right_face_second_to_last >= 0) {

                aU[Nx - 2] = -u_left_face_second_to_last * rho[Nx - 3] - D;
                cU[Nx - 2] = -D;
                bU[Nx - 2] = u_right_face_second_to_last * rho[Nx - 2] + rho[Nx - 2] * dx / dt + 2 * D;
                dU[Nx - 2] = -0.5 * (p[Nx - 1] - p[Nx - 3]) + rho[Nx - 2] * u[Nx - 2] * dx / dt + Su[Nx - 2] * dx;

            }
            else if (u_left_face_second_to_last >= 0 && u_right_face_second_to_last < 0) {

                aU[Nx - 2] = -u_left_face_second_to_last * rho[Nx - 3] - D;
                cU[Nx - 2] = u_right_face_second_to_last * rho[Nx - 1] - D;
                bU[Nx - 2] = rho[Nx - 2] * dx / dt + 2 * D;
                dU[Nx - 2] = -0.5 * (p[Nx - 1] - p[Nx - 3]) + rho[Nx - 2] * u[Nx - 2] * dx / dt + Su[Nx - 2] * dx;

            }
            else if (u_left_face_second_to_last < 0 && u_right_face_second_to_last >= 0) {

                aU[Nx - 2] = -D;
                cU[Nx - 2] = -D;
                bU[Nx - 2] = (u_right_face_second_to_last - u_left_face_second_to_last) * rho[Nx - 2] + rho[Nx - 2] * dx / dt + 2 * D;
                dU[Nx - 2] = -0.5 * (p[Nx - 1] - p[Nx - 3]) + rho[Nx - 2] * u[Nx - 2] * dx / dt + Su[Nx - 2] * dx;

            }
            else if (u_left_face_second_to_last < 0 && u_right_face_second_to_last < 0) {

                aU[Nx - 2] = -D;
                cU[Nx - 2] = u_right_face_second_to_last * rho[Nx - 1] - D;
                bU[Nx - 2] = -u_left_face_second_to_last * rho[Nx - 2] + rho[Nx - 2] * dx / dt + 2 * D;
                dU[Nx - 2] = -0.5 * (p[Nx - 1] - p[Nx - 3]) + rho[Nx - 2] * u[Nx - 2] * dx / dt + Su[Nx - 2] * dx;

            }

            u = solveTridiagonal(aU, bU, cU, dU);

            for (int piso = 0; piso < corr_iter; piso++) {

                std::vector<double> aP(Nx, 0.0), bP(Nx, 0.0), cP(Nx, 0.0), dP(Nx, 0.0);

                // // #pragma omp parallel
                for (int i = 1; i < Nx - 1; i++) {

                    // --- Coefficienti di diffusione di pressione ---

                    // Faccia Ovest (w, tra i-1 e i)
                    double rho_w = 0.5 * (rho[i - 1] + rho[i]);
                    // 1/Ap medio sulla faccia
                    double d_w_face = 0.5 * (1.0 / bU[i - 1] + 1.0 / bU[i]);
                    // Coeff. ellittico (nota: dx*dx)
                    double E_w = rho_w * d_w_face / (dx * dx);

                    // Faccia Est (e, tra i e i+1)
                    double rho_e = 0.5 * (rho[i] + rho[i + 1]);
                    // 1/Ap medio sulla faccia
                    double d_e_face = 0.5 * (1.0 / bU[i] + 1.0 / bU[i + 1]);
                    // Coeff. ellittico
                    double E_e = rho_e * d_e_face / (dx * dx);

                    // --- Compressibilità (psi = d(rho)/d(p)) ---
                    // Assumendo gas ideale: rho = p/(RT) -> d(rho)/d(p) = 1/(RT)
                    double psi_i = 1.0 / (Rv * T[i]); // T è T* (dal passo precedente)

                    // --- Calcolo sbilancio di massa (RHS) ---
                    // (Usa una semplice media per u* e upwind per rho*)
                    double u_w_star = 0.5 * (u[i - 1] + u[i]);
                    double mdot_w_star = (u_w_star > 0.0) ? rho[i - 1] * u_w_star : rho[i] * u_w_star;

                    double u_e_star = 0.5 * (u[i] + u[i + 1]);
                    double mdot_e_star = (u_e_star > 0.0) ? rho[i] * u_e_star : rho[i + 1] * u_e_star;

                    double mass_imbalance = (rho[i] - rho_old[i]) / dt + (mdot_e_star - mdot_w_star) / dx;

                    // --- Assembla Matrice P_PRIME ---
                    aP[i] = -E_w;
                    cP[i] = -E_e;
                    bP[i] = E_w + E_e + psi_i / dt;
                    dP[i] = Sm[i] - mass_imbalance; // S_m - (sbilancio di massa)

                    //double rhie_chow_left = -(1.0 / bU[i - 1] + 1.0 / bU[i]) / (8 * dx) * (p[i - 2] - 3 * p[i - 1] + 3 * p[i] - p[i + 1]);
                    //double rhie_chow_right = -(1.0 / bU[i + 1] + 1.0 / bU[i]) / (8 * dx) * (p[i - 1] - 3 * p[i] + 3 * p[i + 1] - p[i + 2]);

                    //double u_left_face = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_left;
                    //double u_right_face = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_right;

                    //double rhoW = (u_left_face >= 0) ? rho[i - 1] : rho[i];
                    //double rhoE = (u_right_face >= 0) ? rho[i] : rho[i + 1];

                    //double D_W = 0.5 * rhoW * (1.0 / bU[i - 1] + 1.0 / bU[i]) / dx;
                    //double D_E = 0.5 * rhoE * (1.0 / bU[i + 1] + 1.0 / bU[i]) / dx;

                    //double F_W = rhoW * u_left_face;
                    //double F_E = rhoE * u_right_face;

                    //aP[i] = -D_W;
                    //cP[i] = -D_E;
                    //bP[i] = D_E + D_W + rho[i] / p[i] * dx / dt;
                    //dP[i] = F_E - F_W + Sm[i] * dx;

                    printf("");
                }

                // BCs for p'
                // Left: zero gradient correction
                bP[0] = 1.0; cP[0] = -1.0; dP[0] = 0.0;

                // Right: p' = 0 to pin pressure level
                bP[Nx - 1] = 1.0; aP[Nx - 1] = 0.0; dP[Nx - 1] = 0.0;

                //// a_P coefficients for first and last node
                //double bU0 = rho[0] * dx / dt + 2 * D;
                //double bU_Nx_1 = rho[Nx - 1] * dx / dt + 2 * D;

                //// Second node
                //double rhie_chow_left_second = -(1.0 / bU0 + 1.0 / bU[1]) / (8 * dx) * (- 2 * p[0] + 3 * p[1] - p[2]);
                //double rhie_chow_right_second = -(1.0 / bU[2] + 1.0 / bU[1]) / (8 * dx) * (p[0] - 3 * p[1] + 3 * p[2] - p[3]);

                //double u_left_face = 0.5 * (u[0] + u[1]) + rhie_chow_on_off * rhie_chow_left_second;
                //double u_right_face = 0.5 * (u[1] + u[2]) + rhie_chow_on_off * rhie_chow_right_second;

                //double rhoW_second = (u_left_face_second >= 0) ? rho[0] : rho[1];
                //double rhoE_second = (u_right_face_second >= 0) ? rho[1] : rho[2];

                //double D_W_second = 0.5 * rhoW_second * (1.0 / bU0 + 1.0 / bU[1]) / dx;
                //double D_E_second = 0.5 * rhoE_second * (1.0 / bU[2] + 1.0 / bU[1]) / dx;

                //double F_W_second = rhoW_second * u_left_face_second;
                //double F_E_second = rhoE_second * u_right_face_second;

                //aP[1] = -D_W_second;
                //cP[1] = -D_E_second;
                //bP[1] = D_E_second + D_W_second + rho[1] / p[1] * dx / dt;
                //dP[1] = F_E_second - F_W_second + Sm[1] * dx;

                //// Second-to-last node
                //double rhie_chow_left_second_to_last = -(1.0 / bU[Nx - 3] + 1.0 / bU[Nx - 2]) / (8 * dx) * (p[Nx - 4] - 3 * p[Nx - 3] + 3 * p[Nx - 2] - p[Nx - 1]);
                //double rhie_chow_right_second_to_last = -(1.0 / bU_Nx_1 + 1.0 / bU[Nx - 2]) / (8 * dx) * (p[Nx - 3] - 3 * p[Nx - 2] + 2 * p[Nx - 1]);

                //double u_left_face_second_to_last = 0.5 * (u[Nx - 3] + u[Nx - 2]) + rhie_chow_on_off * rhie_chow_left_second_to_last;
                //double u_right_face_second_to_last = 0.5 * (u[Nx - 2] + u[Nx - 1]) + rhie_chow_on_off * rhie_chow_right_second_to_last;

                //double rhoW_second_to_last = (u_left_face_second_to_last >= 0) ? rho[Nx - 3] : rho[Nx - 2];
                //double rhoE_second_to_last = (u_right_face_second_to_last >= 0) ? rho[Nx - 2] : rho[Nx - 1];

                //double D_W_second_to_last = 0.5 * rhoW_second_to_last * (1.0 / bU[Nx - 3] + 1.0 / bU[Nx - 2]) / dx;
                //double D_E_second_to_last = 0.5 * rhoE_second_to_last * (1.0 / bU_Nx_1 + 1.0 / bU[Nx - 2]) / dx;

                //double F_W_second_to_last = rhoW_second_to_last * u_left_face_second_to_last;
                //double F_E_second_to_last = rhoE_second_to_last * u_right_face_second_to_last;

                //aP[Nx - 2] = -D_W_second_to_last;
                //cP[Nx - 2] = -D_E_second_to_last;
                //bP[Nx - 2] = D_E_second_to_last + D_W_second_to_last + rho[Nx - 2] / p[Nx - 2] * dx / dt;
                //dP[Nx - 2] = F_E_second_to_last - F_W_second_to_last + Sm[Nx - 2] * dx;

                p_prime = solveTridiagonal(aP, bP, cP, dP);

                // Correct p and u
                for (int i = 0; i < Nx; i++) p[i] += p_prime[i];

                maxErr = 0.0;
                for (int i = 1; i < Nx - 1; i++) {

                    double u_prev = u[i];
                    u[i] = u[i] - (p_prime[i + 1] - p_prime[i - 1]) / (2.0 * dx * bU[i]);
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

        // #pragma omp parallel
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
