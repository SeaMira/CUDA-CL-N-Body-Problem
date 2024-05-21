#include <chrono>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include "kernel.h"

struct Times {
  long create_data;
  long execution;

  long total() { return create_data + execution; }
};

int pos_x_limit = 100, pos_y_limit = 100, pos_z_limit = 100, vel_x_limit = 100, vel_y_limit = 100, vel_z_limit = 100;
Times t;
float *posiciones;
float *velocidades;

bool simulate(int n) {
  using std::chrono::microseconds;

  posiciones = new float[n*3];
  velocidades = new float[n*3];

  auto t_start = std::chrono::high_resolution_clock::now();



  init_values(pos_x_limit, pos_y_limit, pos_z_limit, posiciones, n);
  init_values(vel_x_limit, vel_y_limit, vel_z_limit, velocidades, n);

  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  t_start = std::chrono::high_resolution_clock::now();
  k_1_times_simulation(posiciones, velocidades, n, 100);
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  std::cout << "RESULTS: " << std::endl;

  std::cout << "Time to create data: " << t.create_data << " microseconds\n";
  std::cout << "Time to execute kernel: " << t.execution << " microseconds\n";
  std::cout << "Time to execute the whole program: " << t.total()
            << " microseconds\n";

  return true;
}

int main(int argc, char* argv[]) {

  if (argc != 3) {
    std::cerr << "Uso: " << argv[0] << " <array size> <output_file>"
              << std::endl;
    return 2;
  }

  int n = std::stoi(argv[1]);
  if (!simulate(n)) {
    std::cerr << "Error while executing the simulation" << std::endl;
    return 3;
  }

  std::ofstream out;
  out.open(argv[2], std::ios::app | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Error while opening file: '" << argv[2] << "'" << std::endl;
    return 4;
  }
  out << n << "," << t.create_data << "," << t.execution << "," << t.total()
      << "\n";

  std::cout << "Data written to " << argv[2] << std::endl;
  delete[] posiciones;
  delete[] velocidades;
  return 0;
}
