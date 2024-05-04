
void vec_sum(int* a, int* b, int* c, int n);

// iniciacion de valores aleatorios en el arreglo arr
void init_values(int x_limit, int y_limit, int z_limit, float *arr, int arr_size);

// obtencion de la aceleracion de un cuerpo en una posicion (x, y, z)
void new_acc(float *pos,  float x, float y, float z, int arr_size, float acc_arr[]);

// actualizacion de la velocidad de un cuerpo
void update_body_vel(float *pos, float *vel, int body_idx, int arr_size, float step);

// actualizacion de la posicion de un cuerpo
void update_body_pos(float *pos, float *vel, int body_idx, int arr_size, float step);

// actualizacion de la velocidad de todos los cuerpos
void update_global_vel(float *pos, float *vel, int body_idx, int arr_size, float step);

// actualizacion de la posicion de todos los cuerpos
void update_global_pos(float *pos, float *vel, int body_idx, int arr_size, float step);

// actualizacion de la velocidad y luego de la posicion de todos los cuerpos
void update(float *pos, float *vel, int body_idx, int arr_size, float step);

// simulacion del problema en pasos "step" del tiempo hasta un entero k
void k_times_simulation(float *pos, float *vel, int arr_size, float step, int k);

// similar a lo anterior pero con step = 1
void k_1_times_simulation(float *pos, float *vel, int arr_size, int k);