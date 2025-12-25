#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

// =======================================================================
//                        TDMA SOLVER
// =======================================================================

std::vector<double> solveTridiagonal(const std::vector<double>& a,
    const std::vector<double>& b,
    const std::vector<double>& c,
    const std::vector<double>& d) {

    const int n = b.size();
    std::vector<double> c_star(n), d_star(n), x(n);

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (int i = 1; i < n; ++i) {
        double m = b[i] - a[i] * c_star[i - 1];
        c_star[i] = c[i] / m;
        d_star[i] = (d[i] - a[i] * d_star[i - 1]) / m;
    }

    x[n - 1] = d_star[n - 1];
    for (int i = n - 2; i >= 0; --i)
        x[i] = d_star[i] - c_star[i] * x[i + 1];

    return x;
}

// =======================================================================
//                        GLOBAL / NUMERICAL SETUP
// =======================================================================

constexpr int    N = 100;
constexpr double L = 1.0;
constexpr double dz = L / N;
constexpr double dt = 1e-3;

constexpr int tot_time_steps = 10000;

constexpr int tot_outer_v = 100;
constexpr int tot_inner_v = 100;

constexpr double outer_tol_v = 1e-6;
constexpr double inner_tol_v = 1e-6;

constexpr double rhie_chow_on_off_v = 1.0;

// =======================================================================
//                        FIELD VARIABLES
// =======================================================================

std::vector<double>
    u_v(N, 0.0),
    u_v_old = u_v,
    T_v_bulk(N, 700.0),
    T_v_bulk_old = T_v_bulk,
    p_v(N, 10000.0),
    p_prime_v(N, 0.0),
    p_storage_v(N + 2, 10000),
    p_v_old = p_v,
    rho_v(N, 0.05),
    rho_v_old = rho_v;

double* p_padded_v = &p_storage_v[1];   /// Pointer to work on the wick pressure padded storage with the same indes

std::vector<double> Gamma(N, 0.0);
std::vector<double> Q(N, 0.0);

constexpr double z_evap_start = 0.0;    // [m]
constexpr double z_evap_end = 0.3;      // [m]
constexpr double z_cond_start = 0.7;    // [m]
constexpr double z_cond_end = 1.0;      // [m]

constexpr double Q_tot = 0.0;          // [W/m3]
constexpr double m_dot = 0.0;          // [kg/m3s]

// constexpr double Q_tot = 7e3;       // [W/m3]
// constexpr double m_dot = 0.01;      // [kg/m3s]

const double Rv = 361.5;
const double mu = 1e-5;
const double k = 0.01;
const double cp = 1000;

// uniform volumetric mass source in evap/cond
const double L_evap = z_evap_end - z_evap_start;
const double L_cond = z_cond_end - z_cond_start;

// BCs
constexpr double u_inlet_v = 0.0;
constexpr double u_outlet_v = 0.0;
// constexpr double p_outlet_v = 10000;

/// The coefficient bVU is needed in momentum predictor loop and pressure correction 
// to estimate the velocities at the faces using the Rhie and Chow correction
std::vector<double>
aVU(N, 0.0),                                        /// Lower tridiagonal coefficient for wick velocity
bVU(N, rho_v[0] * dz / dt
    + 2 * (4.0 / 3.0) * mu / dz),                   /// Central tridiagonal coefficient for wick velocity
    cVU(N, 0.0),                                    /// Upper tridiagonal coefficient for wick velocity
    dVU(N, 0.0);                                    /// Known vector coefficient for wick velocity

std::ofstream v_out("velocity.dat");
std::ofstream p_out("pressure.dat");
std::ofstream T_out("temperature.dat");
std::ofstream rho_out("density.dat");

// =======================================================================
//                                MAIN
// =======================================================================

int main() {

    for (int i = 0; i < N; ++i) p_storage_v[i + 1] = p_v[i];
    p_storage_v[0] = p_v[0];
    p_storage_v[N + 1] = p_v[N - 1];

    for (int i = 0; i < N; ++i) {

        const double z = i * dz;

        if (z >= z_evap_start && z < z_evap_end) {
            Q[i] = Q_tot;
            Gamma[i] = m_dot;
        }
        else if (z > z_cond_start && z <= z_cond_end) {
            Q[i] = -Q_tot;
            Gamma[i] = -m_dot;
        }
    }

    for (int n = 0; n < tot_time_steps; ++n) {

        // Saving old variables
        u_v_old = u_v;
        T_v_bulk_old = T_v_bulk;
        rho_v_old = rho_v;
        p_v_old = p_v;

        double u_error_v = 1.0;
        int outer_v = 0;

        while (outer_v < tot_outer_v && u_error_v > outer_tol_v) {

            // ===========================================================
            // MOMENTUM PREDICTOR
            // ===========================================================

            for (int i = 1; i < N - 1; ++i) {

                const double D_l = (4.0 / 3.0) * mu / dz;       // [kg/(m2s)]
                const double D_r = (4.0 / 3.0) * mu / dz;       // [kg/(m2s)]

                const double avgInvbVU_L = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]); // [m2s/kg]
                const double avgInvbVU_R = 0.5 * (1.0 / bVU[i + 1] + 1.0 / bVU[i]); // [m2s/kg]

                // Rhie–Chow corrections for face velocities
                const double rc_l = -avgInvbVU_L / 4.0 *
                    (p_padded_v[i - 2] - 3.0 * p_padded_v[i - 1] + 3.0 * p_padded_v[i] - p_padded_v[i + 1]); // [m/s]
                const double rc_r = -avgInvbVU_R / 4.0 *
                    (p_padded_v[i - 1] - 3.0 * p_padded_v[i] + 3.0 * p_padded_v[i + 1] - p_padded_v[i + 2]); // [m/s]

                // face velocities (avg + RC)
                const double u_l_face = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rc_l;    // [m/s]
                const double u_r_face = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rc_r;    // [m/s]

                // upwind densities at faces
                const double rho_l = (u_l_face >= 0.0) ? rho_v[i - 1] : rho_v[i];       // [kg/m3]
                const double rho_r = (u_r_face >= 0.0) ? rho_v[i] : rho_v[i + 1];       // [kg/m3]

                const double F_l = rho_l * u_l_face; // [kg/(m2s)]
                const double F_r = rho_r * u_r_face; // [kg/(m2s)]

                aVU[i] =
                    - std::max(F_l, 0.0)
                    - D_l;                                  // [kg/(m2s)]
                cVU[i] =
                    - std::max(-F_r, 0.0)
                    - D_r;                                  // [kg/(m2s)]
                bVU[i] =
                    + std::max(F_r, 0.0) 
                    + std::max(-F_l, 0.0)
                    + rho_v[i] * dz / dt
                    + D_l + D_r;                            // [kg/(m2s)]
                dVU[i] =
                    - 0.5 * (p_v[i + 1] - p_v[i - 1])
                    + rho_v_old[i] * u_v_old[i] * dz / dt;  // [kg/(ms2)]
            }

            /// Diffusion coefficients for the first and last node to define BCs
            const double D_first = (4.0 / 3.0) * mu / dz;
            const double D_last = (4.0 / 3.0) * mu / dz;

            /// Velocity BCs needed variables for the first node
            const double u_r_face_first = 0.5 * (u_v[1]);
            const double rho_r_first = (u_r_face_first >= 0) ? rho_v[0] : rho_v[1];
            const double F_r_first = rho_r_first * u_r_face_first;

            /// Velocity BCs needed variables for the last node
            const double u_l_face_last = 0.5 * (u_v[N - 2]);
            const double rho_l_last = (u_l_face_last >= 0) ? rho_v[N - 2] : rho_v[N - 1];
            const double F_l_last = rho_l_last * u_l_face_last;

            /// Velocity BCs: zero velocity on the first node
            aVU[0] = 0.0;
            bVU[0] = std::max(F_r_first, 0.0) + rho_v[0] * dz / dt + 2 * D_first;
            cVU[0] = 0.0;
            dVU[0] = bVU[0] * u_inlet_v;

            /// Velocity BCs: zero velocity on the last node
            aVU[N - 1] = 0.0;
            bVU[N - 1] = std::max(-F_l_last, 0.0) + rho_v[N - 1] * dz / dt + 2 * D_last;
            cVU[N - 1] = 0.0;
            dVU[N - 1] = bVU[N - 1] * u_outlet_v;

            u_v = solveTridiagonal(aVU, bVU, cVU, dVU);

            double p_error_v = 1.0;
            int inner_v = 0;

            double rho_error_v = 1.0;

            while (inner_v < tot_inner_v && p_error_v > inner_tol_v) {

                // -------------------------------------------------------
                // CONTINUITY SATISFACTOR: assemble pressure correction
                // -------------------------------------------------------

                std::vector<double> aVP(N, 0.0), bVP(N, 0.0), cVP(N, 0.0), dVP(N, 0.0);

                for (int i = 1; i < N - 1; ++i) {

                    const double avgInvbVU_L = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]);     // [m2s/kg]
                    const double avgInvbVU_R = 0.5 * (1.0 / bVU[i + 1] + 1.0 / bVU[i]);     // [m2s/kg]

                    const double rc_l = - avgInvbVU_L / 4.0 *
                        (p_padded_v[i - 2] - 3.0 * p_padded_v[i - 1] + 3.0 * p_padded_v[i] - p_padded_v[i + 1]);    // [m/s]
                    const double rc_r = - avgInvbVU_R / 4.0 *
                        (p_padded_v[i - 1] - 3.0 * p_padded_v[i] + 3.0 * p_padded_v[i + 1] - p_padded_v[i + 2]);    // [m/s]

                    const double psi_i = 1.0 / (Rv * T_v_bulk[i]); // [kg / J]

                    const double u_l_star = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rc_l;    // [m/s]
                    const double u_r_star = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rc_r;    // [m/s]

                    const double Crho_l = u_l_star >= 0 ? (1.0 / (Rv * T_v_bulk[i - 1])) : (1.0 / (Rv * T_v_bulk[i]));  // [s2/m2]
                    const double Crho_r = u_r_star >= 0 ? (1.0 / (Rv * T_v_bulk[i])) : (1.0 / (Rv * T_v_bulk[i + 1]));  // [s2/m2]

                    const double C_l = Crho_l * u_l_star;       // [s/m]
                    const double C_r = Crho_r * u_r_star;       // [s/m]

                    const double rho_l_upwind = (u_l_star >= 0.0) ? rho_v[i - 1] : rho_v[i];    // [kg/m3]
                    const double rho_r_upwind = (u_r_star >= 0.0) ? rho_v[i] : rho_v[i + 1];    // [kg/m3]

                    const double phi_l = rho_l_upwind * u_l_star;   // [kg/(m2s)]
                    const double phi_r = rho_r_upwind * u_r_star;   // [kg/(m2s)]

                    const double mass_imbalance = (phi_r - phi_l) + (rho_v[i] - rho_v_old[i]) * dz / dt;  // [kg/(m2s)]
                    
                    const double mass_flux = Gamma[i] * dz;         // [kg/(m2s)]

                    const double E_l = 0.5 * (rho_v[i - 1] * (1.0 / bVU[i - 1]) + rho_v[i] * (1.0 / bVU[i])) / dz; // [s/m]
                    const double E_r = 0.5 * (rho_v[i] * (1.0 / bVU[i]) + rho_v[i + 1] * (1.0 / bVU[i + 1])) / dz; // [s/m]

                    aVP[i] = 
                        - E_l
                        - std::max(C_l, 0.0)
                        ;               /// [s/m]

                    cVP[i] = 
                        - E_r
                        - std::max(-C_r, 0.0)
                        ;              /// [s/m]

                    bVP[i] = 
                        + E_l + E_r 
                        + std::max(C_r, 0.0)
                        + std::max(-C_l, 0.0)
                        + psi_i * dz / dt;                  /// [s/m]

                    dVP[i] = + mass_flux - mass_imbalance;  /// [kg/(m2s)]

                    printf("");
                }

                // BCs
                aVP[0] = 0.0; 
                bVP[0] = 1.0; 
                cVP[0] = -1.0; 
                dVP[0] = 0.0;

                aVP[N - 1] = - 1.0; 
                bVP[N - 1] = 1.0; 
                cVP[N - 1] = 0.0; 
                dVP[N - 1] = 0.0;

                p_prime_v = solveTridiagonal(aVP, bVP, cVP, dVP);

                // -------------------------------------------------------
                // PRESSURE CORRECTOR
                // -------------------------------------------------------

                p_error_v = 0.0;
                std::vector<double> p_prev = p_v;

                for (int i = 0; i < N; ++i) {
                    p_v[i] += p_prime_v[i];
                    p_storage_v[i + 1] = p_v[i];
                    p_error_v = std::max(p_error_v, std::fabs(p_v[i] - p_prev[i]));
                }

                p_storage_v[0] = p_storage_v[1];
                p_storage_v[N + 1] = p_storage_v[N];

                // -------------------------------------------------------
                // VELOCITY CORRECTOR
                // -------------------------------------------------------

                u_error_v = 0.0;
                std::vector<double> u_prev = u_v;

                for (int i = 1; i < N - 1; ++i) {
                    u_v[i] -= (p_prime_v[i + 1] - p_prime_v[i - 1]) / (2.0 * dz * bVU[i]);
                    u_error_v = std::max(u_error_v, std::fabs(u_v[i] - u_prev[i]));
                }

                // -------------------------------------------------------
                // DENSITY CORRECTOR
                // -------------------------------------------------------

                rho_error_v = 0.0;
                std::vector<double> rho_prev = rho_v;

                for (int i = 0; i < N; ++i) {
                    rho_v[i] += p_prime_v[i] / (Rv * T_v_bulk[i]);
                    rho_error_v = std::max(rho_error_v, std::fabs(rho_v[i] - rho_prev[i]));
                }

                inner_v++;
            }

            outer_v++;
        }

        // ===============================================================
        // TEMPERATURE SOLVER
        // ===============================================================

        // Energy equation for T (implicit), upwind convection, central diffusion
        std::vector<double>
            aVT(N, 0.0),
            bVT(N, 0.0),
            cVT(N, 0.0),
            dVT(N, 0.0);

        for (int i = 1; i < N - 1; i++) {

            const double D_l = k / dz;      /// [W/(m2 K)]
            const double D_r = k / dz;      /// [W/(m2 K)]

            const double avgInvbVU_L = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]);     // [m2s/kg]
            const double avgInvbVU_R = 0.5 * (1.0 / bVU[i + 1] + 1.0 / bVU[i]);     // [m2s/kg]

            const double rc_l = -avgInvbVU_L / 4.0 *
                (p_padded_v[i - 2] - 3.0 * p_padded_v[i - 1] + 3.0 * p_padded_v[i] - p_padded_v[i + 1]);    // [m/s]
            const double rc_r = -avgInvbVU_R / 4.0 *
                (p_padded_v[i - 1] - 3.0 * p_padded_v[i] + 3.0 * p_padded_v[i + 1] - p_padded_v[i + 2]);    // [m/s]

            const double u_l_face = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rc_l;         // [m/s]
            const double u_r_face = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rc_r;         // [m/s]

            const double rho_l = (u_l_face >= 0) ? rho_v[i - 1] : rho_v[i];     // [kg/m3]
            const double rho_r = (u_r_face >= 0) ? rho_v[i] : rho_v[i + 1];     // [kg/m3]
     
            const double Fl = rho_l * u_l_face;         // [kg/m2s]
            const double Fr = rho_r * u_r_face;         // [kg/m2s]
    
            const double C_l = (Fl * cp);               // [W/(m2K)]
            const double C_r = (Fr * cp);               // [W/(m2K)]

            const double dpdz_up = u_v[i] * (p_v[i + 1] - p_v[i - 1]) / 2.0;

            const double dp_dt = (p_v[i] - p_v_old[i]) / dt * dz;

            const double viscous_dissipation =
                4.0 / 3.0 * 0.25 * mu * ((u_v[i + 1] - u_v[i]) * (u_v[i + 1] - u_v[i])
                    + (u_v[i] + u_v[i - 1]) * (u_v[i] + u_v[i - 1])) / dz;

            aVT[i] =
                - D_l
                - std::max(C_l, 0.0);                   /// [W/(m2 K)]

            cVT[i] =
                - D_r
                - std::max(-C_r, 0.0);                  /// [W/(m2 K)]

            bVT[i] =
                + (std::max(C_r, 0.0) + std::max(-C_l, 0.0))
                + D_l + D_r
                + rho_v[i] * cp * dz / dt;              /// [W/(m2 K)]

            dVT[i] =
                + rho_v_old[i] * cp * dz / dt * T_v_bulk_old[i]
                + dp_dt
                + dpdz_up
                + viscous_dissipation
                + Q[i] * dz;                            /// [W/m2]
        }

        /// Temperature BCs: zero gradient on the first node
        aVT[0] = 0.0;
        bVT[0] = 1.0;
        cVT[0] = -1.0;
        dVT[0] = 0.0;

        /// Temperature BCs: zero gradient on the last node
        aVT[N - 1] = -1.0;
        bVT[N - 1] = 1.0;
        cVT[N - 1] = 0.0;
        dVT[N - 1] = 0.0;

        T_v_bulk = solveTridiagonal(aVT, bVT, cVT, dVT);

        for (int i = 0; i < N; i++) { rho_v[i] = std::max(1e-6, p_v[i] / (Rv * T_v_bulk[i])); }

        // ===============================================================
        // OUTPUT
        // ===============================================================

        for (int i = 0; i < N; ++i) {

            v_out << u_v[i] << " ";
            p_out << p_v[i] << " ";
            T_out << T_v_bulk[i] << " ";
            rho_out << rho_v[i] << " ";
        }

        v_out << "\n";
        p_out << "\n";
        T_out << "\n";
        rho_out << "\n";

        v_out.flush();
        p_out.flush();
        T_out.flush();
        rho_out.flush();
    }

    v_out.close();
    p_out.close();
    T_out.close();
    rho_out.flush();

    return 0;
}
