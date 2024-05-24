BUILD=build
PROJECT=MyProject

.PHONY: init clean test run

all: cl cpu cuda

init:
	@cmake -S . -B $(BUILD)

cuda: init
	@cmake --build $(BUILD) --target $(PROJECT)CUDA -j 10

cl: init
	@mkdir -p $(BUILD)/src/cl
	@cp src/cl/kernel.cl $(BUILD)/src/cl/kernel.cl
	@cp src/cl/kernel_wth_local_mem.cl $(BUILD)/src/cl/kernel_wth_local_mem.cl
	@cp src/cl/kernel_bidimensional.cl $(BUILD)/src/cl/kernel_bidimensional.cl
	@cmake --build $(BUILD) --target $(PROJECT)CL -j 10

cpu: init
	@cmake --build $(BUILD) --target $(PROJECT)CPU -j 10

clean:
	rm -rf .cache Testing build
