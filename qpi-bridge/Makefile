NVCC    ?= nvcc
ARCH    ?= -gencode arch=compute_86,code=sm_86
CXXFLAGS = -O3 -march=native -mavx2 -pthread
LDFLAGS  = -lnuma -lpthread

staged_copy: staged_copy.cu
	$(NVCC) $(ARCH) -Xcompiler "$(CXXFLAGS)" -o $@ $< $(LDFLAGS)

clean:
	rm -f staged_copy

.PHONY: clean
