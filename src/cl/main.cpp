
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
#include <filesystem>
#include <cmath>
#include <format>

struct Times {
  long create_data;
  long copy_to_host;
  long execution;
  long copy_to_device;
  inline long total() {
    return create_data + copy_to_host + execution + copy_to_device;
  }
};
cl_ulong localMemSize;
int pos_x_limit = 100, pos_y_limit = 100, pos_z_limit = 100, vel_x_limit = 100, vel_y_limit = 100, vel_z_limit = 100;

Times t;
float *posiciones;
float *velocidades;
cl::Program prog;
cl::CommandQueue queue;

bool init(std::string filename) {
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
  
  cl::Device device = devices.front();

  device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &localMemSize);
  // devices.front().getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(localMemSize)
  std::cout << "Maximum local memory size: " << localMemSize << " bytes" << std::endl;
        

  cl::Context context(devices.front());
  queue = cl::CommandQueue(context, devices.front());

  std::filesystem::path p = std::filesystem::current_path(); // Obtiene la ruta actual

  std::string src_code = load_from_file(std::format("src/cl/{}", filename));
  if (src_code.empty()) src_code = load_from_file(filename);
  // std::cout << src_code << std::endl;
  cl::Program::Sources sources;
  sources.push_back({src_code.c_str(), src_code.length()});

  prog = cl::Program(context, sources);
  auto result = prog.build("-cl-std=CL1.2");
  if (result != CL_SUCCESS) {
      std::string buildlog = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices.front());
      std::cerr << "Error en la compilacion del kernel: " << buildlog << "\n";
      return false;
  }

  return true;
}

bool simulate(int n, int gs, int ls) {
  using std::chrono::microseconds;
  std::size_t size = sizeof(float) * n * 3;

  posiciones = new float[n*3];
  velocidades = new float[n*3];
  // Create the memory buffers
  cl::Buffer posBuff(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);
  cl::Buffer velBuff(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);

  // Assign values to host variables
  auto t_start = std::chrono::high_resolution_clock::now();
  
  init_values(pos_x_limit, pos_y_limit, pos_z_limit, posiciones, n);
  init_values(vel_x_limit, vel_y_limit, vel_z_limit, velocidades, n);

  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();
  
  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  // usar CL_FALSE para hacerlo asíncrono
  int err = queue.enqueueWriteBuffer(posBuff, CL_TRUE, 0, size, posiciones);
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
  kernel.setArg(2, n);
  kernel.setArg(3, step);
  // pasar por argumento la masa, quizás, en un arreglo
  // kernel.setArg(3, N);

  // Execute the function on the device (using 32 threads here)
  cl::NDRange gSize(gs);
  cl::NDRange lSize(ls);

  t_start = std::chrono::high_resolution_clock::now();
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, gSize, lSize);
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

/////////////////////////////////////////////////////////////////////////////////////////////////////
bool simulate_matrix(int n, int gsx, int gsy, int lsx, int lsy) {
  using std::chrono::microseconds;
  std::size_t size = sizeof(float) * n * 3;

  posiciones = new float[n*3];
  velocidades = new float[n*3];
  // Create the memory buffers
  cl::Buffer posBuff(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);
  cl::Buffer velBuff(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);

  // Assign values to host variables
  auto t_start = std::chrono::high_resolution_clock::now();
  
  init_values(pos_x_limit, pos_y_limit, pos_z_limit, posiciones, n);
  init_values(vel_x_limit, vel_y_limit, vel_z_limit, velocidades, n);

  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  // usar CL_FALSE para hacerlo asíncrono
  int err = queue.enqueueWriteBuffer(posBuff, CL_TRUE, 0, size, posiciones);
  queue.enqueueWriteBuffer(velBuff, CL_TRUE, 0, size, velocidades);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_device =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();


  // Make kernel
  cl::Kernel kernel(prog, "bodyInteraction");
  

  float step = 1.0f;
  // int n = pow(2, 6);
  // Set the kernel arguments
  kernel.setArg(0, posBuff);
  kernel.setArg(1, velBuff);
  kernel.setArg(2, n);
  kernel.setArg(3, step);
  kernel.setArg(4, gsx);
  kernel.setArg(5, gsy);

  // pasar por argumento la masa, quizás, en un arreglo
  // kernel.setArg(3, N);

  // Execute the function on the device (using 32 threads here)
  cl::NDRange gSize(gsx, gsy);
  cl::NDRange lSize(lsx, lsy);

  t_start = std::chrono::high_resolution_clock::now();
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, gSize, lSize);
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

long Min(long a, long b) {
  return a < b ? a : b;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
bool simulate_with_local_mem(int n, int gs, int ls, int mem_size) {
  using std::chrono::microseconds;
  std::size_t size = sizeof(float) * n * 3;

  posiciones = new float[n*3];
  velocidades = new float[n*3];
  // Create the memory buffers
  cl::Buffer posBuff(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);
  cl::Buffer velBuff(queue.getInfo<CL_QUEUE_CONTEXT>(), CL_MEM_READ_WRITE, size);

  // Assign values to host variables
  auto t_start = std::chrono::high_resolution_clock::now();
  
  init_values(pos_x_limit, pos_y_limit, pos_z_limit, posiciones, n);
  init_values(vel_x_limit, vel_y_limit, vel_z_limit, velocidades, n);
  // for (int i = 0; i < n; i++) {
  //   std::cout << posiciones[i] << std::endl;
  //   std::cout << posiciones[i+1] << std::endl;
  //   std::cout << posiciones[i+2] << std::endl;
  // }

  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();

  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  // usar CL_FALSE para hacerlo asíncrono
  int err = queue.enqueueWriteBuffer(posBuff, CL_TRUE, 0, size, posiciones);
  queue.enqueueWriteBuffer(velBuff, CL_TRUE, 0, size, velocidades);
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_device =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();


  // Make kernel
  cl::Kernel kernel(prog, "bodyInteraction");
  localMemSize /= sizeof(float)*3;
  std::cout << localMemSize << " localMemSize" << std::endl;
  std::cout << mem_size << " mem_size" << std::endl;
  // mem_size /= sizeof(float)*3;
  localMemSize = Min((long)localMemSize, (long) mem_size);
  std::cout << localMemSize << " minimo" << std::endl;

  float step = 1.0f;
  // Set the kernel arguments
  kernel.setArg(0, posBuff);
  kernel.setArg(1, velBuff);
  kernel.setArg(2, n);
  kernel.setArg(3, step);
  kernel.setArg(4, sizeof(float)*localMemSize*3, NULL);
  kernel.setArg(5, localMemSize);
  // pasar por argumento la masa, quizás, en un arreglo
  // kernel.setArg(3, N);

  // Execute the function on the device (using 32 threads here)
  cl::NDRange gSize(gs);
  cl::NDRange lSize(ls);

  t_start = std::chrono::high_resolution_clock::now();
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, gSize, lSize);
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
  

  if (argc != 9) {
    std::cerr << "Uso: " << argv[0]
              << " <mode> <array size> <local size> <global size> <local size y> <global size y> <local memory size> <output file>"
              << std::endl;
    return 2;
  }
  int mode = std::stoi(argv[1]); // 1- kernel, 2-kernel_with_local_mem, 3-kernel_bidimensional
  int n = std::stoi(argv[2]);
  int ls = std::stoi(argv[3]);
  int gs = std::stoi(argv[4]);
  int lsy = std::stoi(argv[5]);
  int gsy = std::stoi(argv[6]);
  int mem_size = std::stoi(argv[7]);
  std::string out_filename = argv[8];
  // for (int i = 0; i < 8; i++) {
  //   for (int j = 0; j < 3; j++) {
  //     if (!simulate_matrix()) {
  //       std::cerr << "CL: Error while executing the simulation" << std::endl;
  //       return 3;
  //     }
  //   }
  //   bodies*=2;
  // }
  switch (mode)
  {
  case 1:
    if (!init("kernel.cl")) return 1;
    if (!simulate(n, gs, ls)) {
      std::cerr << "CL: Error while executing the simulation" << std::endl;
      return 3;
    }
    break;
  case 2:
    if (!init("kernel_wth_local_mem.cl")) return 1;
    if (!simulate_with_local_mem(n, gs, ls, mem_size)) {
      std::cerr << "CL: Error while executing the simulation" << std::endl;
      return 3;
    }
    break;
  case 3:
    if (!init("kernel_bidimensional.cl")) return 1;
    if (!simulate_matrix(n, gs, gsy, ls, lsy)) {
      std::cerr << "CL: Error while executing the simulation" << std::endl;
      return 3;
    }
    break;
  
  default:
    break;
  }

  std::ofstream out;
  out.open(out_filename, std::ios::app | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Error while opening file: '" << out_filename << "'" << std::endl;
    return 4;
  }
  // params
  out << n << "," << ls << "," << gs << "," << lsy << "," << gsy << "," << localMemSize*3*sizeof(float) << "," << t.create_data << "," << t.copy_to_device << "," << t.execution << ","
      << t.copy_to_host << "\n";
  // times
  std::cout << t.create_data << "," << t.copy_to_device << "," << t.execution << ","
      << t.copy_to_host << "," << t.total() << "\n";

  // std::cout << "Data written to " << argv[4] << std::endl;
  delete[] posiciones;
  delete[] velocidades;
  return 0;
}
