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

constexpr int tot_time_steps = 1000;

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

double* p_padded_v = &p_storage_v[1];           /// Poìnter to work on the wick pressure padded storage with the same indes

std::vector<double> Gamma(N, 0.0);
std::vector<double> Q(N, 0.0);

constexpr double z_evap_start = 0.0;   // [m]
constexpr double z_evap_end = 0.3;     // [m]
constexpr double z_cond_start = 0.7;   // [m]
constexpr double z_cond_end = 1.0;     // [m]

constexpr double Q_tot = 10;           // total heat input [W/m3]
const double m_dot = 0.01;              // [kg/m3s]

const double Rv = 361.5;                /// Gas constant for the sodium vapor [J/(kg K)]
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

/// The coefficient bVU is needed in momentum predictor loop and pressure correction to estimate the velocities at the faces using the Rhie and Chow correction
std::vector<double>
aVU(N, 0.0),                                                /// Lower tridiagonal coefficient for wick velocity
bVU(N, rho_v[0] * dz / dt
    + 2 * mu / dz),                    /// Central tridiagonal coefficient for wick velocity
    cVU(N, 0.0),                                                /// Upper tridiagonal coefficient for wick velocity
    dVU(N, 0.0);                                                /// Known vector coefficient for wick velocity

std::ofstream v_out("velocity.dat");
std::ofstream p_out("pressure.dat");
std::ofstream T_out("temperature.dat");
std::ofstream rho_out("density.dat");

auto rho_correct_from_p = [&](std::vector<double>& rho_,
    const std::vector<double>& p_new,
    const std::vector<double>& p_old,
    const std::vector<double>& T_) {

        for (int i = 0; i < N; ++i) {
            const double psi = 1.0 / (Rv * T_[i]);
            rho_[i] = std::max(1e-6, rho_[i] + psi * (p_new[i] - p_old[i]));
        }
    };

// =======================================================================
//                                MAIN
// =======================================================================

int main() {

    for (int i = 0; i < N; ++i) {

        const double z = i * dz;

        if (z >= z_evap_start && z <= z_evap_end) {
            Q[i] = Q_tot;
            Gamma[i] = m_dot;
        }
        else if (z >= z_cond_start && z <= z_cond_end) {
            Q[i] = -Q_tot;
            Gamma[i] = -m_dot;
        }
    }

    for (int n = 0; n < tot_time_steps; ++n) {

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

                const double rho_P = rho_v[i];
                const double rho_L = rho_v[i - 1];
                const double rho_R = rho_v[i + 1];
                const double rho_P_old = rho_v_old[i];

                const double D_l = mu / dz;
                const double D_r = mu / dz;

                const double invbX_L = 1.0 / bVU[i - 1] + 1.0 / bVU[i];
                const double invbX_R = 1.0 / bVU[i + 1] + 1.0 / bVU[i];

                // Rhie–Chow corrections for face velocities
                const double rc_l = -invbX_L / (8.0 * dz) *
                    (p_padded_v[i - 2] - 3.0 * p_padded_v[i - 1] + 3.0 * p_padded_v[i] - p_padded_v[i + 1]);
                const double rc_r = -invbX_R / (8.0 * dz) *
                    (p_padded_v[i - 1] - 3.0 * p_padded_v[i] + 3.0 * p_padded_v[i + 1] - p_padded_v[i + 2]);

                // face velocities (avg + RC)
                const double u_l_face = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rc_l;
                const double u_r_face = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rc_r;

                // upwind densities at faces
                const double rho_l = (u_l_face >= 0.0) ? rho_L : rho_P;
                const double rho_r = (u_r_face >= 0.0) ? rho_P : rho_R;

                const double F_l = rho_l * u_l_face; // [kg/(m2 s)]
                const double F_r = rho_r * u_r_face; // [kg/(m2 s)]

                const double D_pipe = 0.01;   // diametro idraulico [m]
                const double drag = 32.0 * mu / (D_pipe * D_pipe);  // [kg/(m3 s)]

                aVU[i] =
                    -std::max(F_l, 0.0)
                    - D_l;
                cVU[i] =
                    -std::max(-F_r, 0.0)
                    - D_r;
                bVU[i] =
                    +std::max(F_r, 0.0) + std::max(-F_l, 0.0)
                    + rho_P * dz / dt
                    + D_l + D_r
                    + drag * dz;
                dVU[i] =
                    -0.5 * (p_v[i + 1] - p_v[i - 1])
                    + rho_P_old * u_v_old[i] * dz / dt;
            }

            /// Diffusion coefficients for the first and last node to define BCs
            const double D_first = mu / dz;
            const double D_last = mu / dz;

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
            bVU[N - 1] = -std::max(-F_l_last, 0.0) + rho_v[N - 1] * dz / dt + 2 * D_last;
            cVU[N - 1] = 0.0;
            dVU[N - 1] = bVU[N - 1] * u_outlet_v;

            u_v = solveTridiagonal(aVU, bVU, cVU, dVU);

            double p_error_v = 1.0;
            int inner_v = 0;

            while (inner_v < tot_inner_v && p_error_v > inner_tol_v) {

                // -------------------------------------------------------
                // CONTINUITY SATISFACTOR: assemble pressure correction
                // -------------------------------------------------------

                std::vector<double> aVP(N, 0.0), bVP(N, 0.0), cVP(N, 0.0), dVP(N, 0.0);

                for (int i = 1; i < N - 1; ++i) {

                    const double rho_P = rho_v[i];
                    const double rho_L = rho_v[i - 1];
                    const double rho_R = rho_v[i + 1];

                    const double d_l_face = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]) / dz;
                    const double d_r_face = 0.5 * (1.0 / bVU[i] + 1.0 / bVU[i + 1]) / dz;

                    // RC corrections consistent with your second block
                    const double rc_l = -d_l_face / 4.0 *
                        (p_padded_v[i - 2] - 3.0 * p_padded_v[i - 1] + 3.0 * p_padded_v[i] - p_padded_v[i + 1]);

                    const double rc_r = -d_r_face / 4.0 *
                        (p_padded_v[i - 1] - 3.0 * p_padded_v[i] + 3.0 * p_padded_v[i + 1] - p_padded_v[i + 2]);

                    /// Compressibility [kg/J] assuming ideal gas
                    const double psi_i = 1.0 / (Rv * T_v_bulk[i]);

                    const double u_l_star = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rc_l;
                    const double u_r_star = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rc_r;

                    const double phi_l = (u_l_star > 0.0) ? rho_L * u_l_star : rho_P * u_l_star;
                    const double phi_r = (u_r_star > 0.0) ? rho_P * u_r_star : rho_R * u_r_star;

                    const double mass_imbalance = (phi_r - phi_l);          // [kg/(m2 s)]
                    const double mass_flux = Gamma[i] * dz;    // [kg/(m2 s)] positive out of wick

                    /// Term [kg/(m2 s)] related to the change in density
                    const double change_density = (rho_P - rho_v_old[i]) * dz / dt;

                    const double rho_l = 0.5 * (rho_L + rho_P);
                    const double rho_r = 0.5 * (rho_P + rho_R);

                    const double E_l = rho_l * d_l_face; // [s/m]
                    const double E_r = rho_r * d_r_face; // [s/m]

                    aVP[i] = - E_l;
                    cVP[i] = - E_r;
                    bVP[i] = E_l + E_r + psi_i * dz / dt;  /// [s/m]
                    dVP[i] = + mass_flux - mass_imbalance - change_density;   /// [kg/(m^2 s)]
                    printf("");
                }

                // BCs for p' (same as your second)
                aVP[0] = 0.0; bVP[0] = 1.0; cVP[0] = -1.0; dVP[0] = 0.0;
                aVP[N - 1] = - 1.0; bVP[N - 1] = 1.0; cVP[N - 1] = 0.0; dVP[N - 1] = 0.0;

                p_prime_v = solveTridiagonal(aVP, bVP, cVP, dVP);

                // -------------------------------------------------------
                // PRESSURE CORRECTOR
                // -------------------------------------------------------

                p_error_v = 0.0;

                for (int i = 0; i < N; ++i) {
                    const double p_prev = p_v[i];
                    p_v[i] += p_prime_v[i];
                    p_storage_v[i + 1] = p_v[i];
                    p_error_v = std::max(p_error_v, std::fabs(p_v[i] - p_prev));
                }

                p_storage_v[0] = p_storage_v[1];
                p_storage_v[N + 1] = p_storage_v[N];

                // -------------------------------------------------------
                // VELOCITY CORRECTOR
                // -------------------------------------------------------

                u_error_v = 0.0;
                for (int i = 1; i < N - 1; ++i) {
                    const double u_prev = u_v[i];
                    u_v[i] -= (p_prime_v[i + 1] - p_prime_v[i - 1]) / (2.0 * dz * bVU[i]);
                    u_error_v = std::max(u_error_v, std::fabs(u_v[i] - u_prev));
                }

                inner_v++;

                rho_correct_from_p(rho_v, p_v, p_v_old, T_v_bulk);
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

            /// Physical properties
            const double rho_P = rho_v[i];
            const double rho_L = rho_v[i - 1];
            const double rho_R = rho_v[i + 1];

            /// Diffusion coefficient [W/(m^2 K)] (average)
            const double D_l = k / dz;
            const double D_r = k / dz;

            const double invbVU_L = 1.0 / bVU[i - 1] + 1.0 / bVU[i];
            const double invbVU_R = 1.0 / bVU[i + 1] + 1.0 / bVU[i];

            /// RC correction 
            const double rhie_chow_l = -invbVU_L / (8 * dz) *
                (p_padded_v[i - 2] - 3 * p_padded_v[i - 1] + 3 * p_padded_v[i] - p_padded_v[i + 1]);
            const double rhie_chow_r = -invbVU_R / (8 * dz) *
                (p_padded_v[i - 1] - 3 * p_padded_v[i] + 3 * p_padded_v[i + 1] - p_padded_v[i + 2]);

            /// Face velocity [m/s] (average + RC correction)
            const double u_l_face = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rhie_chow_l;
            const double u_r_face = 0.5 * (u_v[i] + u_v[i + 1]) + rhie_chow_on_off_v * rhie_chow_r;

            /// Density [kg/m3] (upwind)
            const double rho_l = (u_l_face >= 0) ? rho_L : rho_P;
            const double rho_r = (u_r_face >= 0) ? rho_P : rho_R;

            /// Mass flux [kg/(m2 s)] (upwind)       
            const double Fl = rho_l * u_l_face;
            const double Fr = rho_r * u_r_face;

            /// Mass flux [W/(m^2 K)] (upwind)
            const double C_l = (Fl * cp);
            const double C_r = (Fr * cp);

            const double dpdz_up = u_v[i] * (p_v[i + 1] - p_v[i - 1]) / (2.0 * dz);
            const double pressure_work = (p_v[i] - p_v_old[i]) / dt + dpdz_up;
            const double viscous_dissipation =
                0.5 * mu * ((u_v[i + 1] - u_v[i]) * (u_v[i + 1] - u_v[i])
                    + (u_v[i] + u_v[i - 1]) * (u_v[i] + u_v[i - 1])) / (dz * dz);

            aVT[i] =
                - D_l
                - std::max(C_l, 0.0);                   /// [W/(m2 K)]
            cVT[i] =
                - D_r
                - std::max(-C_r, 0.0);                  /// [W/(m2 K)]
            bVT[i] =
                +(std::max(C_r, 0.0) + std::max(-C_l, 0.0))
                + D_l + D_r
                + rho_v[i] * cp * dz / dt;              /// [W/(m2 K)]

            dVT[i] =
                + rho_v_old[i] * cp * dz / dt * T_v_bulk_old[i]
                + pressure_work * dz
                + viscous_dissipation * dz
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
