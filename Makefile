NVCC := nvcc
NVCCFLAGS := -arch=compute_80 -code=compute_80
OBJECTS := sgemm.o cublas_reference.o wmma_kernels.o utils.o

sgemm: $(OBJECTS) 
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) -lcublas -o $@

sgemm.o: sgemm.cu gemm.h utils.h 
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ 

utils.o: utils.cu utils.h gemm.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ 

wmma_kernels.o: wmma_kernels.cu gemm.h utils.h tile_mma.cuh 
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ 

cublas_reference.o: cublas_reference.cu gemm.h 
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ 

.PHONY: clean
clean:
	rm -f sgemm $(OBJECTS) 
  
