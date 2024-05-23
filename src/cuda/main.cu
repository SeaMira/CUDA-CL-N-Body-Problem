#include <cstddef>
#include <chrono>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "kernel.cuh"
#include <bits/getopt_core.h>

struct Times {
  long create_data;
  long copy_to_host;
  long execution;
  long copy_to_device;
  inline long total() {
    return create_data + copy_to_host + execution + copy_to_device;
  }
};
int pos_x_limit = 100, pos_y_limit = 100, pos_z_limit = 100, vel_x_limit = 100, vel_y_limit = 100, vel_z_limit = 100;
int WORK_GROUP_SIZE = 32;
Times t;

void checkCudaErrors(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void init_values(int x_limit, int y_limit, int z_limit, float *arr, int arr_size) {
  // srand(time(0));

  for (int i = 0; i < arr_size*3; i+=3) {
    arr[i] = (float)((rand() % (2*x_limit)) - x_limit) + (float) rand()/RAND_MAX;
    arr[i+1] = (float)((rand() % (2*y_limit)) - y_limit) + (float) rand()/RAND_MAX;
    arr[i+2] = (float)((rand() % (2*z_limit)) - z_limit) + (float) rand()/RAND_MAX;
  }

}
bool simulate(int bodies,int iterations,dim3 block_size,dim3 grid_size,bool localmem) {
  using std::chrono::microseconds;
  std::size_t size = sizeof(float) * bodies * 3;
  std::vector<float> posiciones(bodies*3),velocidades(bodies*3);
  // Create the memory buffers
  int *posBuff, *velBuff;
  cudaMalloc(&posBuff, size);
  cudaMalloc(&velBuff, size);
  // Assign values to host variables
  auto t_start = std::chrono::high_resolution_clock::now();
  init_values(pos_x_limit, pos_y_limit, pos_z_limit, posiciones.data(), bodies);
  init_values(vel_x_limit, vel_y_limit, vel_z_limit, velocidades.data(), bodies);
  auto t_end = std::chrono::high_resolution_clock::now();
  t.create_data =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();
  // Copy values from host variables to device
  t_start = std::chrono::high_resolution_clock::now();
  checkCudaErrors(cudaMemcpy(posBuff, posiciones.data(), size, cudaMemcpyHostToDevice),"pos copy to device");
  checkCudaErrors(cudaMemcpy(velBuff, velocidades.data(), size, cudaMemcpyHostToDevice),"vel copy to device");
  t_end = std::chrono::high_resolution_clock::now();
  t.copy_to_device =
      std::chrono::duration_cast<microseconds>(t_end - t_start).count();
  // Execute the function on the device (using 32 threads here)
  float step = 100.0f/iterations;
  t_start = std::chrono::high_resolution_clock::now();
  if(localmem){
    int sharedMemSize = 3 * block_size.x * sizeof(float);
    bodyInteractionLocal1D<<<grid_size,block_size,sharedMemSize>>>((float*)posBuff,(float*)velBuff, bodies, step);
  }
  else if(grid_size.y!=1 || block_size.y!=1){
    bodyInteraction2D<<<grid_size,block_size>>>((float*)posBuff,(float*)velBuff, bodies, step);
  }
  else{
    bodyInteraction1D<<<grid_size,block_size>>>((float*)posBuff,(float*)velBuff, bodies, step);
  }
  checkCudaErrors(cudaGetLastError(), "Kernel launch");
  checkCudaErrors(cudaDeviceSynchronize(),"sync");
  t_end = std::chrono::high_resolution_clock::now();
  t.execution =
      std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
          .count();
  // Copy the output variable from device to host
  t_start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(posiciones.data(), posBuff, size, cudaMemcpyDeviceToHost);//,"pos copy to host");
  cudaMemcpy(velocidades.data(),velBuff, size, cudaMemcpyDeviceToHost);//,"vel copy to host");
  cudaFree(posBuff);
  cudaFree(velBuff);
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
};


int main(int argc, char* argv[]) {
  bool dim1 = true;
  bool dim2 = false;
  bool localmem = false;
  int opt;
  while ((opt = getopt(argc, argv, "2l")) != -1) {
    switch (opt) {
      case '2':
        dim1= false;
        dim2 = true;
        break;
      case 'l':
        std::cout<<"selected local mem optimization"<<std::endl;
        localmem = true;
        break;
      case '?': // Manejar opciones desconocidas o mal formadas
        break;
        return 0;
    }
  }
  int bodies;
  int iterations;
  dim3 block_size;
  dim3 grid_size;
  std::string file;
  if(dim1){
    if (argc != 6+localmem) {
      std::cerr << "Uso: " << argv[0] << " <array size> <iterations> <block size> <grid size> <output file>"<<std::endl;
      return 2;
    }
    bodies = std::stoi(argv[localmem+1]);
    iterations=std::stoi(argv[localmem+2]);
    block_size =dim3(std::stoi(argv[localmem+3]),1);
    grid_size = dim3(std::stoi(argv[localmem+4]),1);
    file=argv[localmem+5];
  }
  else if(dim2){
    if (argc != 9) {
      std::cerr << "Uso: " << argv[0] << " <array size> <iterations> "
      <<"<block size x> <block size y> <grid size x> <grid size y> <output file>"<<std::endl;
      return 2;
    }
    bodies = std::stoi(argv[2]);
    iterations=std::stoi(argv[3]);
    block_size =dim3(std::stoi(argv[4]),std::stoi(argv[5]));
    grid_size = dim3(std::stoi(argv[6]),std::stoi(argv[7]));
    file=argv[8];
  }
  else{
    return 2;
  }
  if (!simulate(bodies,iterations,block_size,grid_size,localmem)) {
    std::cerr << "CUDA: Error while executing the simulation" << std::endl;
    return 3;
  }
  std::ofstream out;
  out.open(file, std::ios::app | std::ios::out);
  if (!out.is_open()) {
    std::cerr << "Error while opening file: '" << file << "'" << std::endl;
    return 4;
  }
  // params
  out << bodies << ",";
  if (dim2){
    out << block_size.x<<","<<block_size.y << "," << grid_size.x<<","<<grid_size.y << ",";
  }
  else{
    out << block_size.x << "," << grid_size.x << ",";
  }
  // times
  out << t.create_data << "," << t.copy_to_device << "," << t.execution << "," << t.copy_to_host << "," << t.total() << "\n";

  std::cout << "Data written to " << file << std::endl;
  return 0;
}
