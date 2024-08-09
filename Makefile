#### User defined Variables ####

# choose among {-O0 -O1 -O2 -O3}
ADDITIONAL_OPTIM_FALG := -O2
# chose among {"int", "float"}
MATRIX_ELEM_DTYPE := float
# allocated constant memory
CONST_MEM_SIZE := 441

### General Variables ###
SOURCEDIR := src
BUILDDIR := obj
TARGETDIR := bin

### compilers & flags ###
CC := gcc
NVCC := nvcc
# compiler related flags
GCC_FLAGS := -std=c11 -Wall
NVCC_FLAGS := -Xcompiler -Wall -lm
# dynamic libraries
ADDITIONAL_LIBS_C := -lm -lpng
ADDITIONAL_LIBS_CU := -lcurand -lcudart
# reference to headers
INCLUDE := -I$(SOURCEDIR)/headers

# Main executable to test functionalities
BUILDNAME := project-imageProcessing
# File to build statistics for analysis purposes
BUILDBENCH := benchmark
# executable sources
MAIN := main.cu
BENCHMARK := benchmark.cu
# dependancies
OBJECTS := \
	$(BUILDDIR)/matrix.o \
	$(BUILDDIR)/pngUtils.o \
	$(BUILDDIR)/opt_parser.o

GPU_OBJECTS := \
	$(BUILDDIR)/cudaUtils.o \
	$(BUILDDIR)/convolution.o

##
#### Rules ####
##

# All rule builds Both Homework-1 and Homework-2
all: GCC_FLAGS += $(ADDITIONAL_OPTIM_FALG)
all: $(TARGETDIR)/$(BUILDNAME)

benchmark: $(TARGETDIR)/$(BUILDBENCH)

debug: GCC_FLAGS += -DDEBUG -g -gdwarf-2
debug: NVCC_FLAGS += -DDEBUG -G
debug: all
debug: benchmark

##
### build binaries
##

# build intermediate object files with GCC
$(BUILDDIR)/%.o: $(SOURCEDIR)/%.c Makefile
	@mkdir -p $(BUILDDIR) $(TARGETDIR)
	@$(CC) $(GCC_FLAGS) -c $< -o $@ $(ADDITIONAL_LIBS_C) $(INCLUDE) -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)'
	@echo building: $<

# build intermediate object files with NVCC
$(BUILDDIR)/%.o: $(SOURCEDIR)/%.cu Makefile
	@mkdir -p $(BUILDDIR) $(TARGETDIR)
	@$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(ADDITIONAL_LIBS_CU) $(INCLUDE) -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)' -DCONST_MEM_SIZE=$(CONST_MEM_SIZE)
	@echo building: $<

##
### build executables
##

# build GPU main with references
$(TARGETDIR)/$(BUILDNAME): $(SOURCEDIR)/$(MAIN) $(OBJECTS) $(GPU_OBJECTS)
	@mkdir -p $(@D)
	@$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(ADDITIONAL_LIBS_C) $(ADDITIONAL_LIBS_CU) $(INCLUDE) -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)' -DADDITIONAL_OPTIM_FALG='$(ADDITIONAL_OPTIM_FALG)' -DCONST_MEM_SIZE=$(CONST_MEM_SIZE)
	@echo building MAIN into: $@

# build GPU benchmark with references
$(TARGETDIR)/$(BUILDBENCH): $(SOURCEDIR)/$(BENCHMARK) $(OBJECTS) $(GPU_OBJECTS)
	@mkdir -p $(@D)
	@$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(ADDITIONAL_LIBS_C) $(ADDITIONAL_LIBS_CU) $(INCLUDE) -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)' -DADDITIONAL_OPTIM_FALG='$(ADDITIONAL_OPTIM_FALG)' -DCONST_MEM_SIZE=$(CONST_MEM_SIZE)
	@echo building BENCHMARK into: $@

##
### clean rules
##

clean:
	@rm $(BUILDDIR)/*.o $(TARGETDIR)/*
	@rm -r $(TARGETDIR)
	@echo directories cleaned

.PHONY: all debug clean gpu_build
