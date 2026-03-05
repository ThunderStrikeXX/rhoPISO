#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>
#include <filesystem>
#include <chrono>
#include <ctime>

#include "tdma.h"

#pragma region input

namespace fs = std::filesystem;

std::string chooseInputFile(const std::string& inputDir) {
    std::vector<fs::path> files;

    if (!fs::exists(inputDir) || !fs::is_directory(inputDir)) {
        throw std::runtime_error("Input directory not found: " + inputDir);
    }

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }

    if (files.empty()) {
        throw std::runtime_error("No input files found in directory: " + inputDir);
    }

    std::cout << "Available input files:\n";
    for (std::size_t i = 0; i < files.size(); ++i) {
        std::cout << "  [" << i << "] "
            << files[i].filename().string() << '\n';
    }

    std::cout << "Select input file by number: ";
    std::size_t choice;
    std::cin >> choice;

    if (choice >= files.size()) {
        throw std::runtime_error("Invalid selection index");
    }

    return files[choice].string();  // path completo al file scelto
}
          
struct Input {

    int    N = 0;                           // Number of cells [-]
    double L = 0.0;                         // Length of the domain [m]

    double dt_user = 0.0;                   // User-defined time step [s]
    double simulation_time = 0.0;           // Total simulation time [s]

    int    picard_max_iter = 0;             // Maximum Picard iterations [-]
    double picard_tol = 0.0;                // Picard tolerance [-]

    int    piso_outer_iter = 0;             // PISO outer iterations [-]
    int    piso_inner_iter = 0;             // PISO inner iterations [-]
    double piso_outer_tol = 0.0;            // PISO outer tolerance [-]
    double piso_inner_tol = 0.0;            // PISO inner tolerance [-]
    bool   rhie_chow_on_off_v = true;       // Rhie–Chow on/off [-]

    double mu = 0.0;                        // Dynamic viscosity [kg/(m s)]
	double Rv = 0.0;                        // Specific gas constant for water vapor [J/(kg K)]
	double k = 0.0;                         // Thermal conductivity [W/(m K)]
	double cp = 0.0;                        // Specific heat capacity at constant pressure [J/(kg K)]

    double S_m_cell = 0.0;                  // Volumetric mass source [kg/(m3 s)]
    double S_h_cell = 0.0;                  // Volumetric heat source [W/m3]

    double z_evap_start = 0.0;              // Evaporation zone start [m]
    double z_evap_end = 0.0;                // Evaporation zone end [m]
    double z_cond_start = 0.0;              // Condensation zone start [m]
    double z_cond_end = 0.0;                // Condensation zone end [m]

    int    u_inlet_bc = 0;                  // 0 Dirichlet, 1 Neumann
    double u_inlet_value = 0.0;             // [m/s]

    int    u_outlet_bc = 0;                 // 0 Dirichlet, 1 Neumann
    double u_outlet_value = 0.0;            // [m/s]

    int    T_inlet_bc = 0;                  // 0 Dirichlet, 1 Neumann
    double T_inlet_value = 0.0;             // [K]

    int    T_outlet_bc = 0;                 // 0 Dirichlet, 1 Neumann
    double T_outlet_value = 0.0;            // [K]

    int    p_inlet_bc = 0;                  // 0 Dirichlet, 1 Neumann
    double p_inlet_value = 0.0;             // [Pa]

    int    p_outlet_bc = 0;                 // 0 Dirichlet, 1 Neumann
    double p_outlet_value = 0.0;            // [Pa]

    double u_initial = 0.0;                 // [m/s]
    double p_initial = 0.0;                 // [Pa]
    double T_initial = 0.0;                 // [K]
	double rho_initial = 0.0;               // [kg/m3]

    int number_output = 0;                  // Number of outputs [-]

    std::string velocity_file = "";
    std::string pressure_file = "";
    std::string temperature_file = "";
    std::string density_file = "";
};

// =======================================================================
//                                INPUT
// =======================================================================

Input readInput(const std::string& filename) {

    std::ifstream file(filename);
    std::string line, key, eq, value;

    std::unordered_map<std::string, std::string> dict;

    while (std::getline(file, line)) {

        // Removes comments
        auto comment = line.find('#');
        if (comment != std::string::npos)
            line = line.substr(0, comment);

		// Removes empty lines
        if (line.find_first_not_of(" \t") == std::string::npos)
            continue;

        // Finds '='
        auto pos = line.find('=');
        if (pos == std::string::npos)
            continue;

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        // Trim key
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);

        // Trim value
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        dict[key] = value;
    }

    Input in;

    in.N = std::stoi(dict["N"]);
    in.L = std::stod(dict["L"]);

    in.dt_user = std::stod(dict["dt_user"]);
    in.simulation_time = std::stod(dict["simulation_time"]);

    in.piso_outer_iter = std::stoi(dict["piso_outer_iter"]);
    in.piso_inner_iter = std::stoi(dict["piso_inner_iter"]);
    in.piso_outer_tol = std::stod(dict["piso_outer_tol"]);
    in.piso_inner_tol = std::stod(dict["piso_inner_tol"]);
    in.rhie_chow_on_off_v = std::stoi(dict["rhie_chow"]);

    in.mu = std::stod(dict["mu"]);
	in.Rv = std::stod(dict["Rv"]);
	in.k = std::stod(dict["k"]);
	in.cp = std::stod(dict["cp"]);

    in.S_m_cell = std::stod(dict["S_m_cell"]);
    in.S_h_cell = std::stod(dict["S_h_cell"]);

    in.z_evap_start = std::stod(dict["z_evap_start"]);
    in.z_evap_end = std::stod(dict["z_evap_end"]);
    in.z_cond_start = std::stod(dict["z_cond_start"]);
    in.z_cond_end = std::stod(dict["z_cond_end"]);

    in.u_inlet_bc = std::stoi(dict["u_inlet_bc"]);
    in.u_inlet_value = std::stod(dict["u_inlet_value"]);

    in.u_outlet_bc = std::stoi(dict["u_outlet_bc"]);
    in.u_outlet_value = std::stod(dict["u_outlet_value"]);

    in.T_inlet_bc = std::stoi(dict["T_inlet_bc"]);
    in.T_inlet_value = std::stod(dict["T_inlet_value"]);

    in.T_outlet_bc = std::stoi(dict["T_outlet_bc"]);
    in.T_outlet_value = std::stod(dict["T_outlet_value"]);

    in.p_inlet_bc = std::stoi(dict["p_inlet_bc"]);
    in.p_inlet_value = std::stod(dict["p_inlet_value"]);

    in.p_outlet_bc = std::stoi(dict["p_outlet_bc"]);
    in.p_outlet_value = std::stod(dict["p_outlet_value"]);

	in.u_initial = std::stod(dict["u_initial"]);
	in.T_initial = std::stod(dict["T_initial"]);
    in.p_initial = std::stod(dict["p_initial"]);
	in.rho_initial = std::stod(dict["rho_initial"]);

    in.number_output = std::stoi(dict["number_output"]);
    in.velocity_file = dict["velocity_file"];
    in.pressure_file = dict["pressure_file"];
    in.temperature_file = dict["temperature_file"];
	in.density_file = dict["density_file"];

    return in;
}

#pragma endregion

// =======================================================================
//                                MAIN
// =======================================================================

int main() {

    std::string inputFile = chooseInputFile("input");
    std::cout << "Using input file: " << inputFile << std::endl;

    Input in = readInput(inputFile);

	const int    N = in.N;                                              // Number of cells [-]
    const double L = in.L;                                              // Length of the domain [m]
	const double dz = L / N;                                            // Cell size [m]

	double dt_user = in.dt_user;                                        // User-defined time step [s]
	const double simulation_time = in.simulation_time;                  // Total simulation time [s]
	const int time_steps = static_cast<int>(simulation_time / dt_user); // Number of time steps [-]
        
	const int number_output = in.number_output;                         // Number of outputs [-]
	const int print_every = time_steps / number_output;                 // Print output every n time steps [-]

	double time_total = 0.0;                                            // Total simulation time [s]

	const int max_picard = in.picard_max_iter;                          // Maximum Picard iterations [-]
	const double pic_tolerance = in.picard_tol;                         // Picard tolerance [-]

	const int tot_outer_v = in.piso_outer_iter;                         // PISO outer iterations [-]
	const int tot_inner_v = in.piso_inner_iter;                         // PISO inner iterations [-]
	const double outer_tol_v = in.piso_outer_tol;                       // PISO outer tolerance [-]
	const double inner_tol_v = in.piso_inner_tol;                       // PISO inner tolerance [-]
	const bool rhie_chow_on_off_v = in.rhie_chow_on_off_v;              // Rhie–Chow interpolation on/off (1/0) [-]

    double dt = dt_user;                                                // Time step [s]

	const double mu = in.mu;                                            // Dynamic viscosity [kg/(m s)]
	const double Rv = in.Rv;                                            // Specific gas constant for water vapor [J/(kg K)]
	const double k = in.k;                                              // Thermal conductivity [W/(m K)]
	const double cp = in.cp;                                            // Specific heat capacity at constant pressure [J/(kg K)]

	std::vector<double> u_v(N, in.u_initial);                           // Velocity field [m/s]
	std::vector<double> T_v(N, in.T_initial);                           // Temperature field [K]
	std::vector<double> p_v(N, in.p_initial);                           // Pressure field [Pa]
	std::vector<double> rho_v(N, in.rho_initial);                       // Density field [kg/m3]

	std::vector<double> u_v_old = u_v;                                  // Previous time step velocity [m/s]
	std::vector<double> T_v_old = T_v;                                  // Previous time step temperature [K]
	std::vector<double> p_v_old = p_v;                                  // Previous time step pressure [Pa]
	std::vector<double> rho_v_old = rho_v;                              // Previous time step density [kg/m3]

	std::vector<double> p_prime_v(N, 0.0);                              // Pressure correction [Pa]
	std::vector<double> p_storage_v(N + 2, 0.0);                        // Padded pressure storage for Rhie–Chow [Pa]
	double* p_padded_v = &p_storage_v[1];                               // Pointer to the real nodes of the padded pressure storage [Pa]

    std::vector<double> phi_v(N + 1, 0.0);                              // Vapor face mass flux [kg/m2s]

    // p_storp_storage_vage_l initialization
    for (int i = 0; i < N; ++i)
        p_storage_v[i + 1] = p_v[i];

    p_storage_v[0] = p_v[0];
    p_storage_v[N + 1] = p_v[N - 1];

	std::vector<double> u_prev(N, 0.0);                                 // Previous iteration velocity for convergence check [m/s]
	std::vector<double> p_prev(N, 0.0);                                 // Previous iteration pressure for convergence check [Pa]
	std::vector<double> rho_prev(N, 0.0);                               // Previous iteration density for convergence check [kg/m3]
    std::vector<double> T_v_prev(N, 0.0);                               // Previous iteration temperature for convergence check [K]

	std::vector<double> S_m(N, 0.0);                                    // Volumetric mass source [kg/(m3 s)]
	std::vector<double> S_h(N, 0.0);                                    // Volumetric heat source [W/m3]

	// Source vectors definition
    for (int i = 0; i < N; ++i) {

        const double z = (i + 0.5) * dz;

        if (z >= in.z_evap_start && z <= in.z_evap_end) {
            S_m[i] = in.S_m_cell;
            S_h[i] = in.S_h_cell;
        }
        else if (z >= in.z_cond_start && z <= in.z_cond_end) {
            S_m[i] = -in.S_m_cell;
            S_h[i] = -in.S_h_cell;
        }
    }

	const double u_inlet_value = in.u_inlet_value;          // Inlet velocity [m/s]
	const double u_outlet_value = in.u_outlet_value;        // Outlet velocity [m/s]
	const bool u_inlet_bc = in.u_inlet_bc;                  // Inlet velocity BC type (Dirichlet: 0.0, Neumann: 1.0) [-]
	const bool u_outlet_bc = in.u_outlet_bc;                // Outlet velocity BC type (Dirichlet: 0.0, Neumann: 1.0) [-]

	const double T_inlet_value = in.T_inlet_value;          // Inlet temperature [K]
	const double T_outlet_value = in.T_outlet_value;        // Outlet temperature [K]
	const bool T_inlet_bc = in.T_inlet_bc;                  // Inlet temperature BC type (Dirichlet: 0.0, Neumann: 1.0) [-]
	const bool T_outlet_bc = in.T_outlet_bc;                // Outlet temperature BC type (Dirichlet: 0.0, Neumann: 1.0) [-]

	const double p_inlet_value = in.p_inlet_value;          // Inlet pressure [Pa]
	const double p_outlet_value = in.p_outlet_value;        // Outlet pressure [Pa]
	const bool p_inlet_bc = in.p_inlet_bc;                  // Inlet pressure BC type (Dirichlet: 0.0, Neumann: 1.0) [-]
	const bool p_outlet_bc = in.p_outlet_bc;                // Outlet pressure BC type (Dirichlet: 0.0, Neumann: 1.0) [-]

	const double z_evap_start = in.z_evap_start;                        // Evaporation zone start and end [m]
	const double z_evap_end = in.z_evap_end;                            // Evaporation zone start and end [m]

	const double z_cond_start = in.z_cond_start;                        // Evaporation zone start and end [m]
	const double z_cond_end = in.z_cond_end;                            // Condensation zone start and end [m]

	const double L_evap = z_evap_end - z_evap_start;                    // Length of the evaporation zone [m]
	const double L_cond = z_cond_end - z_cond_start;                    // Length of the condensation zone [m]

    std::vector<double> aVU(N, 0.0);                                    // Lower tridiagonal coefficient for velocity
    std::vector<double> bVU(N, rho_v[0] * dz / dt_user + 2 * mu / dz);  // Central tridiagonal coefficient for velocity
    std::vector<double> cVU(N, 0.0);                                    // Upper tridiagonal coefficient for velocity
    std::vector<double> dVU(N, 0.0);                                    // Known vector coefficient for velocity

	std::vector<double> aVP(N, 0.0);                                    // Lower tridiagonal coefficient for pressure
	std::vector<double> bVP(N, 0.0);                                    // Central tridiagonal coefficient for pressure
	std::vector<double> cVP(N, 0.0);                                    // Upper tridiagonal coefficient for pressure
	std::vector<double> dVP(N, 0.0);                                    // Known vector coefficient for pressure

	std::vector<double> aVT(N, 0.0);                                    // Lower tridiagonal coefficient for temperature
	std::vector<double> bVT(N, 0.0);                                    // Central tridiagonal coefficient for temperature
	std::vector<double> cVT(N, 0.0);                                    // Upper tridiagonal coefficient for temperature
	std::vector<double> dVT(N, 0.0);                                    // Known vector coefficient for temperature

    fs::path inputPath(inputFile);
    std::string caseName = inputPath.filename().string();
    fs::path outputDir = fs::path("output") / caseName;
    fs::create_directories(outputDir);

    std::ofstream v_out(outputDir / in.velocity_file);              // Velocity output file
    std::ofstream p_out(outputDir / in.pressure_file);              // Pressure output file
    std::ofstream T_out(outputDir / in.temperature_file);           // Temperature output file
	std::ofstream rho_out(outputDir / in.density_file);             // Density output file

    // TDMA solver
    tdma::Solver tdma_solver(N);

    std::vector<double> h_v(N, in.T_initial * 1000);                    // Vapor enthalpy [J/kg]
    std::vector<double> h_v_old = h_v;

    // PISO Vapor parameters
    const int tot_simple_iter_v = 50;                   // Outer iterations per time-step [-]
    const int tot_piso_iter_v = 10;                     // Inner iterations per outer iteration [-]
    const double momentum_tol_v = 1e-6;              // Tolerance for the outer iterations (velocity) [-]
    const double continuity_tol_v = 1e-6;            // Tolerance for the inner iterations (pressure) [-]
    const double temperature_tol_v = 1e-2;           // Tolerance for the energy equation [-]

    // Residuals for mass, monentum and enthalpy equations
    double momentum_res_v = 1.0;
    double temperature_res_v = 1.0;
    double continuity_res_v = 1.0;

    // Index for vapor outer and inner iterations
    int simple_iter_v = 0;
    int piso_iter_v = 0;

    // Errors for vapor pressure, velocity and density
    double p_error_v = 0.0;
    double u_error_v = 0.0;
    double rho_error_v = 0.0;

    for (int i = 0; i < N; i++) { rho_v[i] = std::max(1e-6, p_v[i] / (Rv * T_v[i])); }

    // Flux initialization
    for (int i = 1; i < N; ++i) {
        const double u_face = 0.5 * (u_v[i - 1] + u_v[i]);
        const double rho_face = (u_face >= 0.0) ? rho_v[i - 1] : rho_v[i];
        phi_v[i] = rho_face * u_face;
    }

    auto wall_start = std::chrono::steady_clock::now();
    std::clock_t cpu_start = std::clock();

	// Time-stepping loop
    for (int n = 0; n <= time_steps; ++n) {

        // Momentum and energy residual initialization to access outer loop
        momentum_res_v = 1.0;
        temperature_res_v = 1.0;

        // Outer iterations reset
        simple_iter_v = 0;

        while ((simple_iter_v < tot_simple_iter_v) && (momentum_res_v > momentum_tol_v || temperature_res_v > temperature_tol_v)) {

            // ===========================================================
            // MOMENTUM PREDICTOR
            // ===========================================================

            for (int i = 1; i < N - 1; ++i) {

                const double D_l = (4.0 / 3.0) * mu / dz;       // [kg/(m2s)]
                const double D_r = (4.0 / 3.0) * mu / dz;       // [kg/(m2s)]

                aVU[i] =
                    - std::max(phi_v[i], 0.0)
                    - D_l;                                  // [kg/(m2s)]
                cVU[i] =
                    - std::max(-phi_v[i + 1], 0.0)
                    - D_r;                                  // [kg/(m2s)]
                bVU[i] =
                    + std::max(phi_v[i + 1], 0.0)
                    + std::max(-phi_v[i], 0.0)
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

            if (u_inlet_bc == 0) {                               // Dirichlet BC
                aVU[0] = 0.0;
                bVU[0] = rho_v[0] * dz / dt + 2 * D_first + F_r_first;
                cVU[0] = 0.0;
                dVU[0] = bVU[0] * u_inlet_value;
            }
            else if (u_inlet_bc == 1) {                          // Neumann BC
                aVU[0] = 0.0;
                bVU[0] = +(rho_v[0] * dz / dt + 2 * D_first + F_r_first);
                cVU[0] = -(rho_v[0] * dz / dt + 2 * D_first + F_r_first);
                dVU[0] = 0.0;
            }

            if (u_outlet_bc == 0) {                             // Dirichlet BC
                aVU[N - 1] = 0.0;
                bVU[N - 1] = +(rho_v[N - 1] * dz / dt + 2 * D_last - F_l_last);
                cVU[N - 1] = 0.0;
                dVU[N - 1] = bVU[N - 1] * u_outlet_value;
            }
            else if (u_outlet_bc == 1) {                        // Neumann BC
                aVU[N - 1] = -(rho_v[N - 1] * dz / dt + 2 * D_last - F_l_last);
                bVU[N - 1] = +(rho_v[N - 1] * dz / dt + 2 * D_last - F_l_last);
                cVU[N - 1] = 0.0;
                dVU[N - 1] = 0.0;
            }

            tdma_solver.solve(aVU, bVU, cVU, dVU, u_v);

            // =========== FLUX CORRECTOR
            #pragma region flux_corrector

            for (int i = 1; i < N; ++i) {

                const double avgInvbVU = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]); // [m2s/kg]

                double rc = -avgInvbVU / 4.0 *
                        (p_padded_v[i - 2] - 3.0 * p_padded_v[i - 1] + 3.0 * p_padded_v[i] - p_padded_v[i + 1]); // [m/s]

                // Face velocities (avg + RC)
                const double u_face = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rc;    // [m/s]

                // Upwind densities at faces
                const double rho = (u_face >= 0.0) ? rho_v[i - 1] : rho_v[i];       // [kg/m3]

                phi_v[i] = rho * u_face;
            }

            #pragma endregion

            // Continuity residual initialization to access inner loop
            continuity_res_v = 1.0;

            // Inner iterations reset
            piso_iter_v = 0;

            while ((piso_iter_v < tot_piso_iter_v) && (continuity_res_v > continuity_tol_v)) {

                // -------------------------------------------------------
                // CONTINUITY SATISFACTOR: assemble pressure correction
                // -------------------------------------------------------

                for (int i = 1; i < N - 1; ++i) {

                    const double psi_i = 1.0 / (Rv * T_v[i]); // [kg/J]

                    const double Crho_l = phi_v[i] >= 0 ? (1.0 / (Rv * T_v[i - 1])) : (1.0 / (Rv * T_v[i]));  // [s2/m2]
                    const double Crho_r = phi_v[i + 1] >= 0 ? (1.0 / (Rv * T_v[i])) : (1.0 / (Rv * T_v[i + 1]));  // [s2/m2]

                    const double rho_l_upwind = (phi_v[i] >= 0.0) ? rho_v[i - 1] : rho_v[i];    // [kg/m3]
                    const double rho_r_upwind = (phi_v[i + 1] >= 0.0) ? rho_v[i] : rho_v[i + 1];    // [kg/m3]

                    const double C_l = Crho_l * phi_v[i] / rho_l_upwind;       // [s/m]
                    const double C_r = Crho_r * phi_v[i + 1] / rho_r_upwind;       // [s/m]

                    const double mass_imbalance = (phi_v[i + 1] - phi_v[i]) + (rho_v[i] - rho_v_old[i]) * dz / dt;  // [kg/(m2s)]

                    const double mass_flux = S_m[i] * dz;         // [kg/(m2s)]

                    const double E_l = 0.5 * (rho_v[i - 1] * (1.0 / bVU[i - 1]) + rho_v[i] * (1.0 / bVU[i])) / dz; // [s/m]
                    const double E_r = 0.5 * (rho_v[i] * (1.0 / bVU[i]) + rho_v[i + 1] * (1.0 / bVU[i + 1])) / dz; // [s/m]

                    aVP[i] =
                        -E_l
                        - std::max(C_l, 0.0)
                        ;               /// [s/m]

                    cVP[i] =
                        -E_r
                        - std::max(-C_r, 0.0)
                        ;              /// [s/m]

                    bVP[i] =
                        +E_l + E_r
                        + std::max(C_r, 0.0)
                        + std::max(-C_l, 0.0)
                        + psi_i * dz / dt;                  /// [s/m]

                    dVP[i] = +mass_flux - mass_imbalance;  /// [kg/(m2s)]
                }

                // BCs on p_prime
                if (p_inlet_bc == 0) {                               // Dirichlet BC
                    aVP[0] = 0.0;
                    bVP[0] = 1.0;
                    cVP[0] = 0.0;
                    dVP[0] = 0.0;
                }
                else if (p_inlet_bc == 1) {                          // Neumann BC
                    aVP[0] = 0.0;
                    bVP[0] = 1.0;
                    cVP[0] = -1.0;
                    dVP[0] = 0.0;
                }

                if (p_outlet_bc == 0) {                              // Dirichlet BC
                    aVP[N - 1] = 0.0;
                    bVP[N - 1] = 1.0;
                    cVP[N - 1] = 0.0;
                    dVP[N - 1] = 0.0;
                }
                else if (p_outlet_bc == 1) {                          // Neumann BC
                    aVP[N - 1] = -1.0;
                    bVP[N - 1] = 1.0;
                    cVP[N - 1] = 0.0;
                    dVP[N - 1] = 0.0;
                }

                tdma_solver.solve(aVP, bVP, cVP, dVP, p_prime_v);

                // -------------------------------------------------------
                // PRESSURE CORRECTOR
                // -------------------------------------------------------

                p_error_v = 0.0;

                for (int i = 0; i < N; ++i) {

                    p_prev[i] = p_v[i];
                    p_v[i] += p_prime_v[i];

                    p_storage_v[i + 1] = p_v[i];
                    p_error_v = std::max(p_error_v, std::fabs(p_v[i] - p_prev[i]));
                }

                // BCs on pressure
                if (p_inlet_bc == 0) {                              // Dirichlet BC

					p_v[0] = p_inlet_value;
                    p_storage_v[0] = p_inlet_value;
                }
                else if (p_inlet_bc == 1) {                         // Neumann BC

					p_v[0] = p_v[1];
                    p_storage_v[0] = p_storage_v[1];
                }

                if (p_outlet_bc == 0) {                              // Dirichlet BC

					p_v[N - 1] = p_outlet_value;
                    p_storage_v[N + 1] = p_outlet_value;
                }
                else if (p_outlet_bc == 1) {                         // Neumann BC

					p_v[N - 1] = p_v[N - 2];
                    p_storage_v[N + 1] = p_storage_v[N];
                }

                // -------------------------------------------------------
                // VELOCITY CORRECTOR
                // -------------------------------------------------------

                u_error_v = 0.0;

                for (int i = 1; i < N - 1; ++i) {
                    u_prev[i] = u_v[i];
                    u_v[i] -= (p_prime_v[i + 1] - p_prime_v[i - 1]) / (2.0 * bVU[i]);
                    u_error_v = std::max(u_error_v, std::fabs(u_v[i] - u_prev[i]));
                }

                // -------------------------------------------------------
                // DENSITY CORRECTOR
                // -------------------------------------------------------

                rho_error_v = 0.0;

                for (int i = 0; i < N; ++i) {
                    rho_prev[i] = rho_v[i];
                    rho_v[i] += p_prime_v[i] / (Rv * T_v[i]);
                    rho_error_v = std::max(rho_error_v, std::fabs(rho_v[i] - rho_prev[i]));
                }

                // =========== FLUX CORRECTOR
                for (int i = 1; i < N; ++i) {

                    const double avgInvbVU = 0.5 * (1.0 / bVU[i - 1] + 1.0 / bVU[i]); // [m2s/kg]

                    double rc = -avgInvbVU / 4.0 *
                        (p_padded_v[i - 2] - 3.0 * p_padded_v[i - 1] + 3.0 * p_padded_v[i] - p_padded_v[i + 1]); // [m/s]

                    // Face velocities (avg + RC)
                    const double u_face = 0.5 * (u_v[i - 1] + u_v[i]) + rhie_chow_on_off_v * rc;    // [m/s]

                    // Upwind densities at faces
                    const double rho = (u_face >= 0.0) ? rho_v[i - 1] : rho_v[i];       // [kg/m3]

                    phi_v[i] = rho * u_face;
                }

                // =========== CONTINUITY RESIDUAL CALCULATOR
                #pragma region continuity_residual_calculator

                continuity_res_v = 0.0;

                for (std::size_t i = 1; i < N - 1; ++i) {

                    continuity_res_v = std::max(continuity_res_v, std::abs(dVP[i]));
                }

                    #pragma endregion

                piso_iter_v++;
            }

            // =========== MOMENTUM RESIDUAL CALCULATOR
            #pragma region momentum_residual_calculator

            momentum_res_v = 0.0;

            for (std::size_t i = 1; i < N - 1; ++i) {
                momentum_res_v = std::max(momentum_res_v, std::abs(aVU[i] * u_v[i - 1] + bVU[i] * u_v[i] + cVU[i] * u_v[i + 1] - dVU[i]));
            }

            #pragma endregion

            // ===============================================================
            // TEMPERATURE SOLVER
            // ===============================================================

            // Energy equation for T (implicit), upwind convection, central diffusion
            for (int i = 1; i < N - 1; i++) {

                const double D_v = k / (cp * dz);      /// [W/(m2 K)]
                const double D_r = k / (cp * dz);      /// [W/(m2 K)]

                const double dpdz_up = u_v[i] * (p_v[i + 1] - p_v[i - 1]) / 2.0;

                const double dp_dt = (p_v[i] - p_v_old[i]) / dt * dz;

                const double viscous_dissipation =
                    4.0 / 3.0 * 0.25 * mu * ((u_v[i + 1] - u_v[i]) * (u_v[i + 1] - u_v[i])
                        + (u_v[i] + u_v[i - 1]) * (u_v[i] + u_v[i - 1])) / dz;

                aVT[i] =
                    - D_v
                    - std::max(phi_v[i], 0.0)
                    ;               /// [W/(m2K)]

                cVT[i] =
                    - D_r
                    - std::max(-phi_v[i + 1], 0.0)
                    ;              /// [W/(m2K)]

                bVT[i] =
                    + std::max(phi_v[i + 1], 0.0)
                    + std::max(-phi_v[i], 0.0)
                    + D_v + D_r
                    + rho_v[i] * dz / dt;          /// [W/(m2 K)]

                dVT[i] =
                    + rho_v_old[i] * dz / dt * h_v_old[i]
                    + dp_dt
                    + dpdz_up
                    + viscous_dissipation
                    + S_h[i] * dz;                      /// [W/m2]
            }

            // BCs on temperature
            if (T_inlet_bc == 0) {                      // Dirichlet BC

                aVT[0] = 0.0;
                bVT[0] = 1.0;
                cVT[0] = 0.0;
                dVT[0] = T_inlet_value * 1000;
            }
            else if (T_inlet_bc == 1) {                 // Neumann BC

                aVT[0] = 0.0;
                bVT[0] = 1.0;
                cVT[0] = -1.0;
                dVT[0] = 0.0;
            }

            if (T_outlet_bc == 0) {                     // Dirichlet BC

                aVT[N - 1] = 0.0;
                bVT[N - 1] = 1.0;
                cVT[N - 1] = 0.0;
                dVT[N - 1] = T_outlet_value * 1000;
            }
            else if (T_outlet_bc == 1) {                // Neumann BC

                aVT[N - 1] = -1.0;
                bVT[N - 1] = 1.0;
                cVT[N - 1] = 0.0;
                dVT[N - 1] = 0.0;
            }

            T_v_prev = T_v;
            tdma_solver.solve(aVT, bVT, cVT, dVT, h_v);

            // Recovering temperture from enthalpy
            for (std::size_t i = 0; i < N; i++) {

                T_v[i] = h_v[i] / 1000;

            }

            // =========== TEMPERATURE RESIDUAL CALCULATOR
            #pragma region temperature_residual_calculator

            temperature_res_v = 0.0;

            for (std::size_t i = 0; i < N; ++i) {

                temperature_res_v = std::max(
                    temperature_res_v,
                    std::abs(T_v[i] - T_v_prev[i]) / T_v_prev[i]
                );
            }

            #pragma endregion

            for (int i = 0; i < N; i++) { rho_v[i] = std::max(1e-6, p_v[i] / (Rv * T_v[i])); }

            simple_iter_v++;
        }

        for (int i = 0; i < N; i++) { rho_v[i] = std::max(1e-6, p_v[i] / (Rv * T_v[i])); }

        // Saving old variables
        u_v_old = u_v;
        p_v_old = p_v;
        rho_v_old = rho_v;
        T_v_old = T_v;
        h_v_old = h_v;

        // ===============================================================
        // OUTPUT
        // ===============================================================

        if (n % print_every == 0) {
            for (int i = 0; i < N; ++i) {

                v_out << u_v[i] << ", ";
                p_out << p_v[i] << ", ";
                T_out << T_v[i] << ", ";
				rho_out << rho_v[i] << ", ";
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
    }

    v_out.close();
    p_out.close();
    T_out.close();
	rho_out.close();

    auto wall_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> wall_elapsed = wall_end - wall_start;
    std::cout << "Wall clock time: " << wall_elapsed.count() << " s\n";

    std::clock_t cpu_end = std::clock();
    std::cout << "CPU time: " << (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC << " s\n";

    system("pause");

    return 0;
}