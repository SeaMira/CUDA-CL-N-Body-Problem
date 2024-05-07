# CUDA-CL Template

Este proyecto es un _template_ que les permitirá guiarse sobre como trabajar con CUDA y OpenCL en CMake. Se incluye código básico para la implementación de un proyecto de suma de vectores, la idea es crear los ejecutables para CPU, CUDA y OpenCL, de tal forma que estos se puedan manipular lo suficiente como para despues ser usados mediante Python para crear distintos experimentos. En este caso, cada programa guarda los datos de interés en un CSV. Puede revisar la carpeta `experiment` para ver como utilizarlos en conjunto.

Puede hacer uso de este _template_ haciendo un fork.

## Makefile

Hay un Makefile para trabajar más fácil con los siguientes comandos:

- all: Construye los ejecutables para CUDA, OpenCL y CPU.
- init: Inicializa el directorio de `build` utilizando CMake.
- cuda: Construye el ejecutable para CUDA.
- cl: Construye el ejecutable para OpenCL.
- cpu: Construye el ejecutable para CPU.
- test: Ejecuta las pruebas utilizando CTest.
- clean: Elimina los artefactos de construcción y los directorios de pruebas.
- watch: Monitorea los archivos fuente en busca de cambios y desencadena una construcción cuando se detecta un cambio.
