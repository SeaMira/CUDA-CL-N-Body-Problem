
#include <cstddef>
#include "auxilliary.h"
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif  // DEBUG
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

struct Times {
  long create_data;
  long copy_to_host;
  long execution;
  long copy_to_device;
  inline long total() {
    return create_data + copy_to_host + execution + copy_to_device;
  }
};
int bodies = 516, pos_x_limit = 100, pos_y_limit = 100, pos_z_limit = 100, vel_x_limit = 100, vel_y_limit = 100, vel_z_limit = 100;
int WORK_GROUP_SIZE = 32;
Times t;
float *posiciones;
float *velocidades;
cl::Program prog;
cl::CommandQueue queue;

bool init() {
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;
  cl::Platform::get(&platforms);
  for (auto& p : platforms) {
    p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() > 0) break;
  }
  if (devices.size() == 0) {
    std::cerr << "Not GPU device found" << std::endl;
    return false;
  }

  std::cout << "GPU Used: " << devices.front().getInfo<CL_DEVICE_NAME>()
            << std::endl;

  cl::Context context(devices.front());
  queue = cl::CommandQueue(context, devices.front());

  std::ifstream sourceFile("kernel.cl");
  std::stringstream sourceCode;
  sourceCode << sourceFile.rdbuf();

  prog = cl::Program(context, sourceCode.str(), true);

  return true;
}

bool simulate() {
  using std::chrono::microseconds;
  std::size_t size = sizeof(float) * bodies * 3;

  posiciones = new float[bodies*3];
  velocidades = new float[bodies*3];

  // Create the memory buffers
  cl::Buffer posBuff(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);
  cl::Buffer velBuff(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);

  // Assign values to host variables
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

  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  // usar CL_FALSE para hacerlo asíncrono
  queue.enqueueWriteBuffer(posBuff, CL_TRUE, 0, size, posiciones);
  queue.enqueueWriteBuffer(velBuff, CL_TRUE, 0, size, velocidades);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_device =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // Make kernel
  cl::Kernel kernel(prog, "bodyInteraction");

  float step = 1.0f;
  // Set the kernel arguments
  kernel.setArg(0, posBuff);
  kernel.setArg(1, velBuff);
  kernel.setArg(2, bodies);
  kernel.setArg(3, step);
  // pasar por argumento la masa, quizás, en un arreglo
  // kernel.setArg(3, N);

  // Execute the function on the device (using 32 threads here)
  cl::NDRange gSize((bodies+WORK_GROUP_SIZE-1)/WORK_GROUP_SIZE);
  cl::NDRange lSize(WORK_GROUP_SIZE);

  t_start = std::chrono::high_resolution_clock::now();
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, (bodies+WORK_GROUP_SIZE-1)/WORK_GROUP_SIZE, WORK_GROUP_SIZE);
  queue.finish();
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Copy the output variable from device to host
  t_start = std::chrono::high_resolution_clock::now();
  queue.enqueueReadBuffer(posBuff, CL_TRUE, 0, size, posiciones);
  queue.enqueueReadBuffer(velBuff, CL_TRUE, 0, size, velocidades);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_host =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();

  // Print the result
  std::cout << "RESULTS: " << std::endl;
  // for (int i = 0; i < N; i++)
  //   std::cout << "  out[" << i << "]: " << c[i] << " (" << a[i] << " + " << b[i]
  //             << ")\n";

  std::cout << "Time to create data: " << t.create_data << " microseconds\n";
  std::cout << "Time to copy data to device: " << t.copy_to_device
            << " microseconds\n";
  std::cout << "Time to execute kernel: " << t.execution << " microseconds\n";
  std::cout << "Time to copy data to host: " << t.copy_to_host
            << " microseconds\n";
  std::cout << "Time to execute the whole program: " << t.total()
            << " microseconds\n";
  return true;
}

int main(int argc, char* argv[]) {
  if (!init()) return 1;

  // if (argc != 5) {
  //   std::cerr << "Uso: " << argv[0]
  //             << " <array size> <local size> <global size> <output file>"
  //             << std::endl;
  //   return 2;
  // }
  // int n = std::stoi(argv[1]);
  // int ls = std::stoi(argv[2]);
  // int gs = std::stoi(argv[3]);

  if (!simulate()) {
    std::cerr << "CL: Error while executing the simulation" << std::endl;
    return 3;
  }

  // std::ofstream out;
  // out.open(argv[4], std::ios::app | std::ios::out);
  // if (!out.is_open()) {
  //   std::cerr << "Error while opening file: '" << argv[2] << "'" << std::endl;
  //   return 4;
  // }
  // // params
  // out << n << "," << ls << "," << gs << ",";
  // times
  std::cout << t.create_data << "," << t.copy_to_device << "," << t.execution << ","
      << t.copy_to_host << "," << t.total() << "\n";

  // std::cout << "Data written to " << argv[4] << std::endl;
  delete[] posiciones;
  delete[] velocidades;
  return 0;
}
