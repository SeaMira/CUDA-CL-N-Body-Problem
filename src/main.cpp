#include <chrono>
#include <fstream>
#include <vector>
#include <iostream>
#include "kernel.h"

struct Times {
  long create_data;
  long execution;

  long total() { return create_data + execution; }
};

int bodies = 516, pos_x_limit = 100, pos_y_limit = 100, pos_z_limit = 100, vel_x_limit = 100, vel_y_limit = 100, vel_z_limit = 100;
Times t;
float *posiciones;
float *velocidades;

bool simulate() {
  using std::chrono::microseconds;
  // std::vector<int> a(N), b(N), c(N);

  posiciones = new float[bodies*3];
  velocidades = new float[bodies*3];

  auto t_start = std::chrono::high_resolution_clock::now();

  // for (int i = 0; i < N; i++) {
  //   a[i] = std::rand() % 2000;
  //   b[i] = std::rand() % 2000;
  //   c[i] = 0;
  // }

  init_values(pos_x_limit, pos_y_limit, pos_z_limit, posiciones, bodies);
  init_values(vel_x_limit, vel_y_limit, vel_z_limit, velocidades, bodies);

  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  t_start = std::chrono::high_resolution_clock::now();
  // vec_sum(a.data(), b.data(), c.data(), N);
  k_1_times_simulation(posiciones, velocidades, bodies, 100);
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Print the result
  std::cout << "RESULTS: " << std::endl;
  // for (int i = 0; i < N; i++)
  //   std::cout << "  out[" << i << "]: " << c[i] << " (" << a[i] << " + " << b[i]
  //             << ")\n";

  std::cout << "Time to create data: " << t.create_data << " microseconds\n";
  std::cout << "Time to execute kernel: " << t.execution << " microseconds\n";
  std::cout << "Time to execute the whole program: " << t.total()
            << " microseconds\n";

  return true;
}

int main(int argc, char* argv[]) {
  // if (argc != 3) {
  //   std::cerr << "Uso: " << argv[0] << " <array size> <output_file>"
  //             << std::endl;
  //   return 2;
  // }

  // int n = std::stoi(argv[1]);
  for (int i = 0; i < 6; i++) {
    if (!simulate()) {
      std::cerr << "Error while executing the simulation" << std::endl;
      return 3;
    }
    bodies *= 2;
  }

  // std::ofstream out;
  // out.open(argv[2], std::ios::app | std::ios::out);
  // if (!out.is_open()) {
  //   std::cerr << "Error while opening file: '" << argv[2] << "'" << std::endl;
  //   return 4;
  // }
  // out << "," << t.create_data << "," << t.execution << "," << t.total()
  //     << "\n";

  // std::cout << "Data written to " << argv[2] << std::endl;
  delete[] posiciones;
  delete[] velocidades;
  return 0;
}
