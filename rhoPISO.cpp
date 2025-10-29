#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <algorithm>
#include <array>

// =======================================================================
//
//                        [SOLVING ALGORITHMS]
//
// =======================================================================

#pragma region solver

// Solves a tridiagonal system Ax = d using the Thomas algorithm
// a, b, c are the sub-diagonal, main diagonal, and super-diagonal of A
// d is the r-hand side vector 
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

// =======================================================================
//
//                       [MATERIAL PROPERTIES]
//
// =======================================================================

#pragma region liquid_sodium_properties

/**
 * @brief Provides thermophysical properties for Liquid Sodium (Na).
 *
 * This namespace contains constant data and functions to calculate key
 * temperature-dependent properties of liquid sodium, which is commonly used
 * as a coolant in fast breeder reactors.
 * * All functions accept temperature T in **Kelvin [K]** and return values
 * in standard SI units.
 */
namespace liquid_sodium {

    // Critical temperature [K]
    constexpr double Tcrit = 2509.46;

    // Density [kg/m^3]
    double rho(double T) { return 219.0 + 275.32 * (1.0 - T / Tcrit) + 511.58 * pow(1.0 - T / Tcrit, 0.5); }

    // Thermal conductivity [W/(m·K)]
    double k(double T) { return 124.67 - 0.11381 * T + 5.5226e-5 * T * T - 1.1842e-8 * T * T * T; }

    // Specific heat [J/(kg·K)]
    double cp(double T) {
        double dVT = T - 273.15;
        return 1436.72 - 0.58 * dVT + 4.627e-4 * dVT * dVT;
    }

    // Dynamic viscosity [Pa·s] using Shpilrain et al. correlation, valid for 371 K < T < 2500 K
    double mu(double T) { return std::exp(-6.4406 - 0.3958 * std::log(T) + 556.835 / T); }
}

#pragma endregion

#pragma region vapor_sodium_properties

/**
 * @brief Provides thermophysical and transport properties for Saturated Sodium Vapor.
 *
 * This namespace contains constant data and functions to calculate key properties
 * of sodium vapor, particularly focusing on its behavior near the saturation curve
 * and critical region. It includes functions for thermodynamic properties and
 * flow/heat transfer correlations.
 *
 * All functions primarily accept temperature T in **Kelvin [K]** and return values
 * in standard SI units unless otherwise noted.
 */

namespace vapor_sodium {

    constexpr double Tcrit_Na = 2509.46;           // Critical temperature [K]
    constexpr double Ad_Na = 3.46;                 // Adiabatic factor [-]
    constexpr double m_g_Na = 23e-3;               // Molar mass [kg/mol]

    // Functions that clamps a value x to the range [a, b]
    inline double clamp(double x, double a, double b) { return std::max(a, std::min(x, b)); }

    // 1D table interpolation in T over monotone grid
    template<size_t N>
    double interp_T(const std::array<double, N>& Tgrid, const std::array<double, N>& Ygrid, double T) {

        if (T <= Tgrid.front()) return Ygrid.front();
        if (T >= Tgrid.back())  return Ygrid.back();

        // locate interval
        size_t i = 0;
        while (i + 1 < N && !(Tgrid[i] <= T && T <= Tgrid[i + 1])) ++i;

        return Ygrid[i] + (T - Tgrid[i]) / (Tgrid[i + 1] - Tgrid[i]) * (Ygrid[i + 1] - Ygrid[i]);
    }

    // Enthalpy of vaporization [J/kg]
    inline double h_vap(double T) {

        const double r = 1.0 - T / Tcrit_Na;
        return (393.37 * r + 4398.6 * std::pow(r, 0.29302)) * 1e3;
    }

    // Saturation pressure [Pa]
    inline double P_sat(double T) {

        const double val_MPa = std::exp(11.9463 - 12633.7 / T - 0.4672 * std::log(T));
        return val_MPa * 1e6;
    }

    // Derivative of saturation pressure with respect to temperature [Pa/K]
    inline double dP_sat_dVT(double T) {

        const double val_MPa_per_K =
            (12633.73 / (T * T) - 0.4672 / T) * std::exp(11.9463 - 12633.73 / T - 0.4672 * std::log(T));
        return val_MPa_per_K * 1e6;
    }

    // Density of saturated vapor [kg/m^3]
    inline double rho(double T) {

        const double hv = h_vap(T);                         // [J/kg]
        const double dPdVT = dP_sat_dVT(T);                   // [Pa/K]
        const double rhol = liquid_sodium::rho(T);          // [kg/m^3]
        const double denom = hv / (T * dPdVT) + 1.0 / rhol;
        return 1.0 / denom;                                 // [kg/m^3]
    }

    // Specific heats from table interpolation [J/kg/K]
    inline double cp(double T) {

        static const std::array<double, 21> Tgrid = { 400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400 };
        static const std::array<double, 21> Cpgrid = { 860,1250,1800,2280,2590,2720,2700,2620,2510,2430,2390,2360,2340,2410,2460,2530,2660,2910,3400,4470,8030 };

        // Table also lists 2500 K = 417030; extreme near critical. If needed, extend:
        if (T >= 2500.0) return 417030.0;

        return interp_T(Tgrid, Cpgrid, T);
    }

    inline double cv(double T) {

        static const std::array<double, 21> Tgrid = { 400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400 };

        static const std::array<double, 21> Cvgrid = { 490, 840,1310,1710,1930,1980,1920,1810,1680,1580,1510,1440,1390,1380,1360,1330,1300,1300,1340,1440,1760 };

        // Table also lists 2500 K = 17030; extreme near critical. If needed, extend:
        if (T >= 2500.0) return 17030.0;

        return interp_T(Tgrid, Cvgrid, T);
    }

    // Dynamic viscosity of sodium vapor [Pa·s]
    inline double mu(double T) { return 6.083e-9 * T + 1.2606e-5; }

    /**
     * @brief Calculates the Thermal Conductivity (k) of sodium vapor over an extended range.
     *
     * Performs bilinear interpolation inside the experimental grid.
     * Outside 900–1500 K or 981–98066 Pa, it extrapolates using kinetic-gas scaling (k ∝ sqrt(T))
     * referenced to the nearest boundary. Prints warnings when extrapolating.
     *
     * @param T Temperature [K]
     * @param P Pressure [Pa]
     * @return double Thermal conductivity [W/(m·K)]
     */
    inline double k(double T, double P) {

        static const std::array<double, 7> Tgrid = { 900,1000,1100,1200,1300,1400,1500 };
        static const std::array<double, 5> Pgrid = { 981,4903,9807,49033,98066 };

        static const double Ktbl[7][5] = {
            // P = 981,   4903,    9807,    49033,   98066  [Pa]
            {0.035796, 0.0379,  0.0392,  0.0415,  0.0422},   // 900 K
            {0.034053, 0.043583,0.049627,0.0511,  0.0520},   // 1000 K
            {0.036029, 0.039399,0.043002,0.060900,0.0620},   // 1100 K
            {0.039051, 0.040445,0.042189,0.052881,0.061133}, // 1200 K
            {0.042189, 0.042886,0.043816,0.049859,0.055554}, // 1300 K
            {0.045443, 0.045908,0.046373,0.049859,0.054508}, // 1400 K
            {0.048930, 0.049162,0.049511,0.051603,0.054043}  // 1500 K
        };

        // local clamp
        auto clamp_val = [](double x, double minv, double maxv) {
            return (x < minv) ? minv : ((x > maxv) ? maxv : x);
            };

        auto idz = [](double x, const auto& grid) {
            size_t i = 0;
            while (i + 1 < grid.size() && x > grid[i + 1]) ++i;
            return i;
            };

        const double Tmin = Tgrid.front(), Tmax = Tgrid.back();
        const double Pmin = Pgrid.front(), Pmax = Pgrid.back();

        bool Tlow = (T < Tmin);
        bool Thigh = (T > Tmax);
        bool Plow = (P < Pmin);
        bool Phigh = (P > Pmax);

        double Tc = clamp_val(T, Tmin, Tmax);
        double Pc = clamp_val(P, Pmin, Pmax);

        const size_t iT = idz(Tc, Tgrid);
        const size_t iP = idz(Pc, Pgrid);

        const double T0 = Tgrid[iT], T1 = Tgrid[std::min(iT + 1ul, Tgrid.size() - 1)];
        const double P0 = Pgrid[iP], P1 = Pgrid[std::min(iP + 1ul, Pgrid.size() - 1)];

        const double q11 = Ktbl[iT][iP];
        const double q21 = Ktbl[std::min(iT + 1ul, Tgrid.size() - 1)][iP];
        const double q12 = Ktbl[iT][std::min(iP + 1ul, Pgrid.size() - 1)];
        const double q22 = Ktbl[std::min(iT + 1ul, Tgrid.size() - 1)][std::min(iP + 1ul, Pgrid.size() - 1)];

        double k_interp = 0.0;

        // bilinear interpolation
        if ((T1 != T0) && (P1 != P0)) {
            const double t = (Tc - T0) / (T1 - T0);
            const double u = (Pc - P0) / (P1 - P0);
            k_interp = (1 - t) * (1 - u) * q11 + t * (1 - u) * q21 + (1 - t) * u * q12 + t * u * q22;
        }
        else if (T1 != T0) {
            const double t = (Tc - T0) / (T1 - T0);
            k_interp = q11 + t * (q21 - q11);
        }
        else if (P1 != P0) {
            const double u = (Pc - P0) / (P1 - P0);
            k_interp = q11 + u * (q12 - q11);
        }
        else {
            k_interp = q11;
        }

        // extrapolation handling
        if (Tlow || Thigh || Plow || Phigh) {
            if (Tlow)
                std::cerr << "[Warning] Sodium vapor k(): T=" << T << " < " << Tmin << " K. Using sqrt(T) extrapolation.\n";
            if (Thigh)
                std::cerr << "[Warning] Sodium vapor k(): T=" << T << " > " << Tmax << " K. Using sqrt(T) extrapolation.\n";
            if (Plow || Phigh)
                std::cerr << "[Warning] Sodium vapor k(): P outside ["
                << Pmin << "," << Pmax << "] Pa. Using constant-P approximation.\n";

            double Tref = (Tlow ? Tmin : (Thigh ? Tmax : Tc));
            double k_ref = k_interp;
            double k_extrap = k_ref * std::sqrt(T / Tref);
            return k_extrap;
        }

        return k_interp;
    }


    // Friction factor [-] (Gnielinski correlation)
    inline double f(double Re) {

        if (Re <= 0.0) throw std::invalid_argument("Error: Re < 0");

        const double t = 0.79 * std::log(Re) - 1.64;
        return 1.0 / (t * t);
    }

    // Nusselt number [-] (Gnielinski correlation)
    inline double Nu(double Re, double Pr) {

        // If laminar, Nu is constant
        if (Re < 1000) return 4.36;

        if (Re <= 0.0 || Pr <= 0.0) throw std::invalid_argument("Error: Re or Pr < 0");

        const double f = vapor_sodium::f(Re);
        const double fp8 = f / 8.0;
        const double num = fp8 * (Re - 1000.0) * Pr;
        const double den = 1.0 + 12.7 * std::sqrt(fp8) * (std::cbrt(Pr * Pr) - 1.0); // Pr^(2/3)
        return num / den;
    }

    // Convective heat transfer coefficient [W/m^2/K] (Gnielinski correlation)
    inline double h_conv(double Re, double Pr, double k, double Dh) {
        if (k <= 0.0 || Dh <= 0.0) throw std::invalid_argument("k, Dh > 0");
        const double Nu = vapor_sodium::Nu(Re, Pr);
        return Nu * k / Dh;
    }
}

#pragma endregion

int main() {

    // =======================================================================
    //
    //                      [CONSTANTS AND VARIABLES]
    //
    // =======================================================================

    #pragma region constant_variables

    // Geometric parameters
    const double L = 1.0;                       // Length of the domain [m]
    const int    N = 100;                       // Number of nodes [-]
    const double dz = L / N;                    // Grid spacing [m]
    const double D_pipe = 0.1;                  // Pipe diameter [m], used only to estimate Reynolds number

    // Time-stepping parameters
    const double dt = 0.001;                                // Timestep [s]
    const double t_max = 1.0;                               // Time interval [s]
    const int t_iter = (int)std::round(t_max / dt);         // Number of timesteps [-]

    // PISO parameters
    const int tot_iter = 200;                   // Inner iterations per step [-]
    const int corr_iter = 2;                    // PISO correctors per iteration [-]
    const double tol = 1e-8;                    // Tolerance for the inner iterations [-]

    // Physical properties
    const double Rv = 361.8;                    // Gas constant for the sodium vapor [J/(kg K)]
    const double T_init = 1000;

    // Fields
    std::vector<double> u(N, 0.01), p(N, 50000.0), T(N, T_init), rho(N, 0.5);        // Collocated grid, values in center-cell
    std::vector<double> p_storage(N + 2, 50000.0);                                  // Storage for ghost nodes aVT the boundaries
    double* p_padded = &p_storage[1];                                               // Poìnter to work on the storage with the same indes
    std::vector<double> T_old(N, T_init), rho_old(N, 0.5), p_old(N, 50000.0);        // Backup values
    std::vector<double> p_prime(N, 0.0);                                            // Pressure correction

    // Boundary conditions (Dirichlet p aVT outlet, T aVT both ends, u inlet)
    const double u_inlet = 0.0;         // Inlet velocity [m/s]
    const double u_outlet = 0.0;        // Outlet velocity [m/s]
    const double p_outlet = 50000.0;    // Outlet pressure [Pa]

    // Update equation of state
    auto eos_update = [&](std::vector<double>& rho_, const std::vector<double>& p_, const std::vector<double>& T_) {
    #pragma omp parallel
        for (int i = 0; i < N; i++) {
            double Ti = std::max(200.0, T_[i]);
            rho_[i] = std::max(1e-6, p_[i] / (Rv * Ti));
        }
    }; eos_update(rho, p, T);

    // Mass source and sink definitions
    std::vector<double> Sm(N, 0.0);

    const double mass_source_zone = 0.2;
    const double mass_sink_zone = 0.2;

    const double mass_source_nodes = std::floor(N * mass_source_zone);
    const double mass_sink_nodes = std::floor(N * mass_sink_zone);

    for (int ix = 1; ix < N - 1; ++ix) {
        
        if (ix > 0 && ix <= mass_source_nodes) Sm[ix] = 0.1;
        else if (ix >= (N - mass_sink_nodes) && ix < (N - 1)) Sm[ix] = -0.1;

    }

    // Momentum source
    std::vector<double> Su(N, 0.0);

    // Energy source
    std::vector<double> St(N, 0.0);

    const double energy_source_zone = 0.2;
    const double energy_sink_zone = 0.2;

    const double energy_source_nodes = std::floor(N * energy_source_zone);
    const double energy_sink_nodes = std::floor(N * energy_sink_zone);

    for (int ix = 1; ix < N - 1; ++ix) {

        if (ix > 0 && ix <= energy_source_nodes) St[ix] = 500000.0;
        else if (ix >= (N - energy_sink_nodes) && ix < (N - 1)) St[ix] = -500000.0;

    }

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
    const double Pr_t = 0.01;                           // Prandtl turbulent number for sodium vapor [-]

    // Turbulence fields for sodium vapor initialization
    std::vector<double> k_turb(N, k0);
    std::vector<double> omega_turb(N, omega0);
    std::vector<double> mu_t(N, 0.0);

    // Models
    const int rhie_chow_on_off = 1;                 // 0: no RC correction, 1: with RC correction
    const int SST_model_turbulence_on_off = 0;      // 0: no turbulence, 1: with turbulence

    // The coefficient bVU is needed in momentum predictor loop and pressure correction to estimate the velocities aVT the faces using the Rhie and Chow correction
    std::vector<double> aVU(N, 0.0), bVU(N, 2 * (4.0 / 3.0 * vapor_sodium::mu(T_init) / dz) + dz / dt * rho[0]), cVU(N, 0.0), dVU(N, 0.0);

    #pragma endregion

    // Output file
    std::ofstream fout("solution_rhoPISO.txt");

    // Number of processors available for parallelization
    printf("Threads: %d\n", omp_get_max_threads());

    for (double it = 0; it < t_iter; it++) {

        const double max_u = *std::max_element(u.begin(), u.end());
        const double max_rho = *std::max_element(rho.begin(), rho.end());
        const double min_T = *std::min_element(T.begin(), T.end());

        std::cout << "Solving! Time elapsed:" << dt * it << "/" << t_max 
                        << ", max courant number: " << max_u * dt / dz
                        << ", max reynolds number: " << max_u * D_pipe * max_rho / vapor_sodium::mu(min_T) << "\n";

        // Backup variables
        T_old = T;
        rho_old = rho;
        p_old = p;

        // PISO iterations
        double maxErr = 1.0;
        int iter = 0;

        while (iter<tot_iter && maxErr>tol) {

            // =======================================================================
            //
            //                      [MOMENTUM PREDICTOR]
            //
            // =======================================================================

            #pragma region momentum_predictor

            #pragma omp parallel
            for (int i = 1; i < N - 1; i++) {

                const double rho_P = rho[i];
                const double rho_L = rho[i - 1];
                const double rho_R = rho[i + 1];

                const double mu_P = vapor_sodium::mu(T[i]);
                const double mu_L = vapor_sodium::mu(T[i - 1]);
                const double mu_R = vapor_sodium::mu(T[i + 1]);

                const double D_l = 4.0 / 3.0 * 0.5 * (mu_P + mu_L) / dz;
                const double D_r = 4.0 / 3.0 * 0.5 * (mu_P + mu_R) / dz;

                const double rhie_chow_l = - (1.0 / bVU[i - 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded[i - 2] - 3 * p_padded[i - 1] + 3 * p_padded[i] - p_padded[i + 1]);
                const double rhie_chow_r = - (1.0 / bVU[i + 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded[i - 1] - 3 * p_padded[i] + 3 * p_padded[i + 1] - p_padded[i + 2]);

                const double u_l_face = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_l;
                const double u_r_face = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_r;

                const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;
                const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;

                const double F_l = rho_l * u_l_face;
                const double F_r = rho_r * u_r_face;

                aVU[i] = -std::max(F_l, 0.0) - D_l;
                cVU[i] = std::max(-F_r, 0.0) - D_r;
                bVU[i] = (std::max(F_r, 0.0) - std::max(-F_l, 0.0)) + rho_P * dz / dt + D_l + D_r;
                dVU[i] = -0.5 * (p[i + 1] - p[i - 1]) + rho_P * u[i] * dz / dt + Su[i] * dz;
            }

            // Velocity BC: Dirichlet aVT l, dirichlet aVT r
            const double D_first = 4.0 / 3.0 * vapor_sodium::mu(T[0]) / dz;
            const double D_last = 4.0 / 3.0 * vapor_sodium::mu(T[N - 1]) / dz;

            bVU[0] = rho[0] * dz / dt + 2 * D_first; cVU[0] = 0.0; dVU[0] = (rho[0] * dz / dt + 2 * D_first) * u_inlet;
            aVU[N - 1] = 0.0; bVU[N - 1] = rho[N - 1] * dz / dt + 2 * D_last; dVU[N - 1] = (rho[N - 1] * dz / dt + 2 * D_last) * u_outlet;

            u = solveTridiagonal(aVU, bVU, cVU, dVU);

            #pragma endregion

            for (int piso = 0; piso < corr_iter; piso++) {

                // =======================================================================
                //
                //                        [MASS SATISFACTOR]
                //
                // =======================================================================

                #pragma region pressure_corrector

                std::vector<double> aP(N, 0.0), bP(N, 0.0), cP(N, 0.0), dP(N, 0.0);

                #pragma omp parallel
                for (int i = 1; i < N - 1; i++) {

                    const double rho_P = rho[i];
                    const double rho_L = rho[i - 1];
                    const double rho_R = rho[i + 1];

                    const double rhie_chow_l = -(1.0 / bVU[i - 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded[i - 2] - 3 * p_padded[i - 1] + 3 * p_padded[i] - p_padded[i + 1]);
                    const double rhie_chow_r = -(1.0 / bVU[i + 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded[i - 1] - 3 * p_padded[i] + 3 * p_padded[i + 1] - p_padded[i + 2]);

                    const double rho_w = 0.5 * (rho[i - 1] + rho[i]);
                    const double d_w_face = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]); // 1/Ap average on west face
                    const double E_l = rho_w * d_w_face / dz;

                    const double rho_e = 0.5 * (rho[i] + rho[i + 1]);
                    const double d_e_face = 0.5 * (1.0 / bVU[i] + 1.0 / bVU[i + 1]);  // 1/Ap average on east face
                    const double E_r = rho_e * d_e_face / dz;

                    const double psi_i = 1.0 / (Rv * T[i]); // Compressibility assuming ideal gas

                    const double u_w_star = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_l;
                    const double mdot_w_star = (u_w_star > 0.0) ? rho_L * u_w_star : rho_P * u_w_star;

                    const double u_e_star = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_r;
                    const double mdot_e_star = (u_e_star > 0.0) ? rho_P * u_e_star : rho_R * u_e_star;

                    const double mass_imbalance = (rho_P - rho_old[i]) * dz / dt + (mdot_e_star - mdot_w_star);

                    aP[i] = -E_l;
                    cP[i] = -E_r;
                    bP[i] = E_l + E_r + psi_i * dz / dt;
                    dP[i] = Sm[i] * dz - mass_imbalance;

                    printf("");
                }

                // BCs for p': zero gradient aVT inlet and zero correction aVT outlet
                bP[0] = 1.0; cP[0] = -1.0; dP[0] = 0.0;
                bP[N - 1] = 1.0; aP[N - 1] = 0.0; dP[N - 1] = 0.0;

                p_prime = solveTridiagonal(aP, bP, cP, dP);

                #pragma endregion

                // =======================================================================
                //
                //                        [PRESSURE CORRECTOR]
                //
                // =======================================================================

                #pragma region pressure_updater

                for (int i = 0; i < N; i++) {

                    p[i] += p_prime[i]; // Note that PISO does not require an under-relaxation factor
                    p_storage[i + 1] = p[i];

                }

                p_storage[0] = p_storage[1];
                p_storage[N + 1] = p_outlet;

                #pragma endregion

                // =======================================================================
                //
                //                        [VELOCITY CORRECTOR]
                //
                // =======================================================================

                #pragma region velocity_updater

                maxErr = 0.0;
                for (int i = 1; i < N - 1; i++) {

                    double u_prev = u[i];
                    u[i] = u[i] - (p_prime[i + 1] - p_prime[i - 1]) / (2.0 * dz * bVU[i]);
                    maxErr = std::max(maxErr, std::fabs(u[i] - u_prev));
                }

                #pragma endregion

            }

            iter++;
        }

        // Update density with new p,T
        eos_update(rho, p, T);

        // =======================================================================
        //
        //                        [TURBULENCE MODELIZATION]
        //
        // =======================================================================

        #pragma region turbulence_SST

        // TODO: check discretization scheme

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

                double mu = vapor_sodium::mu(T[i]);
                double mu_eff = mu + mu_t[i];
                double Dw = mu_eff / (sigma_k * dz * dz);
                double De = mu_eff / (sigma_k * dz * dz);
                       aK[i] = -Dw;
                cK[i] = -De;
                bK[i] = rho[i] / dt + Dw + De + beta_star * rho[i] * omega_turb[i];
                dK[i] = rho[i] / dt * k_turb[i] + Pk[i];
            }
        
            // k BCs: constant initial values aVT the boundaries
            bK[0] = 1.0; dK[0] = k_turb[0]; cK[0] = 0.0;
            aK[N - 1] = 0.0; bK[N - 1] = 1.0; dK[N - 1] = k_turb[N - 1];

            k_turb = solveTridiagonal(aK, bK, cK, dK);

            // --- omega-equation ---
            for (int i = 1; i < N - 1; i++) {

                double mu = vapor_sodium::mu(T[i]);
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

                double mu = vapor_sodium::mu(T[i]);
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
        std::vector<double> aVT(N, 0.0), bVT(N, 0.0), cVT(N, 0.0), dVT(N, 0.0);

        #pragma omp parallel
        for (int i = 1; i < N - 1; i++) {

            const double rho_P = rho[i];
            const double rho_L = rho[i - 1];
            const double rho_R = rho[i + 1];

            const double k_cond_P = vapor_sodium::k(T[i], p[i]);
            const double k_cond_L = vapor_sodium::k(T[i - 1], p[i - 1]);
            const double k_cond_R = vapor_sodium::k(T[i + 1], p[i + 1]);

            const double cp_P = vapor_sodium::cp(T[i]);
            const double cp_L = vapor_sodium::cp(T[i - 1]);
            const double cp_R = vapor_sodium::cp(T[i + 1]);

            const double rhoCp_dzdt = rho_old[i] * cp_P * dz / dt;

            const double keff_P = k_cond_P + SST_model_turbulence_on_off * (mu_t[i] * cp_P / Pr_t);
            const double keff_L = k_cond_L + SST_model_turbulence_on_off * (mu_t[i - 1] * cp_L / Pr_t);
            const double keff_R = k_cond_R + SST_model_turbulence_on_off * (mu_t[i + 1] * cp_R / Pr_t);

            // Linear interpolation diffusion coefficient
            const double D_l = 0.5 * (keff_P + keff_L) / dz;
            const double D_r = 0.5 * (keff_P + keff_R) / dz;

            const double rhie_chow_l = -(1.0 / bVU[i - 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded[i - 2] - 3 * p_padded[i - 1] + 3 * p_padded[i] - p_padded[i + 1]);
            const double rhie_chow_r = -(1.0 / bVU[i + 1] + 1.0 / bVU[i]) / (8 * dz) * (p_padded[i - 1] - 3 * p_padded[i] + 3 * p_padded[i + 1] - p_padded[i + 2]);

            const double u_l_face = 0.5 * (u[i - 1] + u[i]) + rhie_chow_on_off * rhie_chow_l;
            const double u_r_face = 0.5 * (u[i] + u[i + 1]) + rhie_chow_on_off * rhie_chow_r;

            // Upwind density
            const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;
            const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;

            // Upwind specific heat
            const double cp_l = (u_l_face >= 0) ? cp_L : cp_P;
            const double cp_r = (u_r_face >= 0) ? cp_P : cp_R;

            const double Fl = rho_l * u_l_face;
            const double Fr = rho_r * u_r_face;

            const double C_l = (Fl * cp_l);
            const double C_r = (Fr * cp_r);

            aVT[i] = -D_l - std::max(C_l, 0.0);
            cVT[i] = -D_r + std::max(-C_r, 0.0);
            bVT[i] = (std::max(C_r, 0.0) - std::max(-C_l, 0.0)) + D_l + D_r + rhoCp_dzdt;

            const double pressure_work = (p[i] - p_old[i]) / dt;
            dVT[i] = rhoCp_dzdt * T_old[i] + pressure_work * dz + St[i] * dz;

        }

        // Temperature BCs
        bVT[0] = 1.0; cVT[0] = -1.0; dVT[0] = 0.0;
        aVT[N - 1] = -1.0; bVT[N - 1] = 1.0; dVT[N - 1] = 0.0;

        T = solveTridiagonal(aVT, bVT, cVT, dVT);

        // Update density with new p,T
        eos_update(rho, p, T);

        #pragma endregion

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
