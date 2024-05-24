# Análisis de NBody-problem con CUDA y OpenCL

Este proyecto presenta los archivos necesarios para ejecutar simulaciones del problema de N cuerpos en su versión secuencial en CPU y su versión en paralelo utilizando CUDA y OpenCL, las simulaciónes realizan 100 iteraciónes con particulas de posiciones y velocidades iniciales aleatorias. Para CPU registra el tiempo de creación de los valores y el tiempo de ejecución, mientras que para las ejecuciones en GPU registra ademas los tamaños de los bloques y grillas utilizados, asi como los tiempos de traspaso de datos hacia la GPU y viseversa. Los datos son posteriormente guardados en el archivo que se le proporcione al ejecutar el programa.

## Compilar el proyecto

Hay un Makefile para trabajar más fácil con los siguientes comandos:

- all: Construye los ejecutables para CUDA, OpenCL y CPU.
- init: Inicializa el directorio de `build` utilizando CMake.
- cuda: Construye el ejecutable para CUDA.
- cl: Construye el ejecutable para OpenCL.
- cpu: Construye el ejecutable para CPU.
- clean: Elimina los artefactos de construcción y los directorios de pruebas.

Los archivos ejecutables se encuentran en el directorio `build/src`, dentro de esta el archivo `MyprojectCPU` contiene la ejecución secuencial del problema, para ejecutarlo se requiere entregar la cantidad de particulas a simular y una ruta a un archivo donde guardar los resultados, de no existir el archivo el programa lo crea.
En la subcarpeta `cl` se encuentra la implementación en OpenCl llamada `MyprojectCL`, para ejecutar es necesario especificar un optCode que define el tipo de ejecución:
- 1 : define la ejecución en paralelo de una dimensión.
- 2 : define la ejecución de una dimensión con memoria compartida
- 3 : define la ejecución en dos dimensiónes.
Luego se recibe la cantidad de particulas
Los siguientes argumentos definen los tamaños locales y globales de la ejecución, se recibe en orden: local_size_x, global_size_x, local_size_y, global_size_y
Se recibe el tamaño de la memoria compartida a usar, este debe ser suficiente para cubrir 3 veces el tamaño local de la ejecución, solo se toma en cuenta si es que se selecciona la ejecucion local.
Finalmente se recibe el archivo donde guardar los resultados.
"<mode> <array size> <local size> <global size> <local size y> <global size y> <local memory size> <output file>"
En la subcarpeta `cuda` tenemos la implementacion `MyprojectCUDA`, a diferencia de cl  el optCode es opcional
- sin optcode : se recibe en orden: cantidad de particulas, block_size, grid_size, ruta al archivo de resultados.
- optcode -l : ejecuta la versión con memoria compartida, recibe los mismos argumentos anteriores, no es necesario especificar el tamaño de memoria, pues se calcula con el tamaño del bloque
- optocde -2 : ejecuta la version en dos dimensiones, recibe: cantidad de particulas, block_size_x, block_size_y, grid_size_x, grid_size_y, ruta al archivo.
-l (optional) <array size> <block size> <grid size> <output file>
-2 <array size> <block size x> <block size y> <grid size x> <grid size y> <output file>
## Ejecutar los experimentos


