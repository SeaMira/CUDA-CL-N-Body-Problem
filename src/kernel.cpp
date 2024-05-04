#include "kernel.h"
#include <cstdlib> 
#include <ctime> 
#include <iostream>
#include <cmath>

using namespace std;

// No se sabe si usar las constantes, o si utilizar masa variable
float G = 1;
float M = 1;

void vec_sum(int* a, int* b, int* c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

// generador de valores flotantes aleatorios en un arreglo donde cada elemento toma 3 espacios: ej, el elemento i tiene componentes en arr[i], arr[i+1], arr[i+2]
void init_values(int x_limit, int y_limit, int z_limit, float *arr, int arr_size) {
  srand(time(0));

  for (int i = 0; i < arr_size*3; i+=3) {
    arr[i] = (float)((rand() % (2*x_limit)) - x_limit) + (float) rand()/RAND_MAX;
    arr[i+1] = (float)((rand() % (2*y_limit)) - y_limit) + (float) rand()/RAND_MAX;
    arr[i+2] = (float)((rand() % (2*z_limit)) - z_limit) + (float) rand()/RAND_MAX;
  }

}


void new_acc(float *pos,  float x, float y, float z, int arr_size, float acc_arr[]) {
  float x_com = 0, y_com = 0, z_com = 0, dx, dy, dz, dist, cubedDist, vec;

  for (int i = 0; i < arr_size*3; i+=3) {
    dx = pos[i] - x;
    dy = pos[i+1] - y;
    dz = pos[i+2] - z;

    dist = sqrt(dx*dx + dy*dy + dz*dz);
    if (dist != 0) {
      cubedDist = dist*dist*dist;
      vec = M/cubedDist;

      x_com += vec*dx;
      y_com += vec*dy;
      z_com += vec*dz;
    }
  }
  acc_arr[0] = x_com;
  acc_arr[1] = y_com;
  acc_arr[2] = z_com;
}

// actualziación de velocidad de un cuerpo, ver si en este caso se necesita un step (en caso de visualización, se delimita por los fps)
void update_body_vel(float *pos, float *vel, int body_idx, int arr_size, float step) {
  float acc[3];
  new_acc(pos, pos[body_idx], pos[body_idx+1], pos[body_idx+2], arr_size, acc);
  vel[body_idx] += acc[0] * step;
  vel[body_idx+1] += acc[1] * step;
  vel[body_idx+2] += acc[2]* step;
}

// actualización de posición de un cuerpo
void update_body_pos(float *pos, float *vel, int body_idx, int arr_size, float step) {
  pos[body_idx] += vel[0] * step;
  pos[body_idx+1] += vel[1] * step;
  pos[body_idx+2] += vel[2]* step;
}

void update_global_vel(float *pos, float *vel, int arr_size, float step) {
  for (int i = 0; i < arr_size*3; i+=3) 
  update_body_vel(pos, vel, i, arr_size, step);
}

void update_global_pos(float *pos, float *vel, int arr_size, float step) {
  for (int i = 0; i < arr_size*3; i+=3) 
  update_body_pos(pos, vel, i, arr_size, step);
}

void update(float *pos, float *vel, int arr_size, float step) {
  update_global_vel(pos, vel, arr_size, step);
  update_global_pos(pos, vel, arr_size, step);
}

void k_times_simulation(float *pos, float *vel, int arr_size, float step, int k) {
  float dt = 0;
  while (dt < k) {
    update(pos, vel, arr_size, step);
    dt += step;
  }
}

void k_1_times_simulation(float *pos, float *vel, int arr_size, int k) {
  float dt = 0;
  while (dt < k) {
    update(pos, vel, arr_size, 1.0f);
    dt += 1.0f;
  }
}