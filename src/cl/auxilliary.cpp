#include <cstdlib> 
#include <ctime> 

using namespace std;

// generador de valores flotantes aleatorios en un arreglo donde cada elemento toma 3 espacios: ej, el elemento i tiene componentes en arr[i], arr[i+1], arr[i+2]
void init_values(int x_limit, int y_limit, int z_limit, float *arr, int arr_size) {
  srand(time(0));

  for (int i = 0; i < arr_size*3; i+=3) {
    arr[i] = (float)((rand() % (2*x_limit)) - x_limit) + (float) rand()/RAND_MAX;
    arr[i+1] = (float)((rand() % (2*y_limit)) - y_limit) + (float) rand()/RAND_MAX;
    arr[i+2] = (float)((rand() % (2*z_limit)) - z_limit) + (float) rand()/RAND_MAX;
  }

}