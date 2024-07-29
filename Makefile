#### User defined Variables ####

# choose among {-O0 -O1 -O2 -O3}
ADDITIONAL_OPTIM_FALG := -O2
# chose among {"int", "float"}
MATRIX_ELEM_DTYPE := float

#### General Variables ####
SOURCEDIR := src
BUILDDIR := obj
TARGETDIR := bin

CC := gcc
NVCC := nvcc

INCLUDE := -I$(SOURCEDIR)/headers

OPT := -std=c11 -Wall -lm 
NVCC_FLAGS := -Xcompiler -Wall -lm -lcurand

# Main executable to test function
BUILDNAME := project-imageProcessing
# File to build CPU statistics for analysis purposes
BUILDBENCH := benchmark

# main file
MAIN := main.cu
BENCHMARK := benchmark.cu

OBJECTS := \
	$(BUILDDIR)/matrix.o \
	$(BUILDDIR)/opt_parser.o

GPU_OBJECTS := \
	$(BUILDDIR)/cudaUtils.o \
	$(BUILDDIR)/convolution.o

##
#### Rules ####
##

# All rule builds Both Homework-1 and Homework-2
all: OPT += $(ADDITIONAL_OPTIM_FALG)
all: $(TARGETDIR)/$(BUILDNAME)

benchmark: $(TARGETDIR)/$(BUILDBENCH)

debug: OPT += -DDEBUG -g
debug: NVCC_FLAGS += -DDEBUG -G 
debug: all
debug: benchmark

##
### build binaries
##

# build intermediate object files with GCC
$(BUILDDIR)/%.o: $(SOURCEDIR)/%.c Makefile
	@mkdir -p $(BUILDDIR) $(TARGETDIR)
	@$(CC) -c -o $@ $(INCLUDE) $< $(OPT) -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)'
	@echo building: $<

# build intermediate object files with NVCC
$(BUILDDIR)/%.o: $(SOURCEDIR)/%.cu Makefile
	@mkdir -p $(BUILDDIR) $(TARGETDIR)
	@$(NVCC) -c -o $@ $(INCLUDE) $(NVCC_FLAGS) $< -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)'
	@echo building: $<

##
### build executables
##

# build GPU main with references
$(TARGETDIR)/$(BUILDNAME): $(SOURCEDIR)/$(MAIN) $(OBJECTS) $(GPU_OBJECTS)
	@mkdir -p $(@D)
	@$(NVCC) $^ -o $@ $(INCLUDE) $(NVCC_FLAGS) -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)' -DADDITIONAL_OPTIM_FALG='$(ADDITIONAL_OPTIM_FALG)'
	@echo building MAIN into: $@

# build GPU benchmark with references
$(TARGETDIR)/$(BUILDBENCH): $(SOURCEDIR)/$(BENCHMARK) $(OBJECTS) $(GPU_OBJECTS)
	@mkdir -p $(@D)
	@$(NVCC) $^ -o $@ $(INCLUDE) $(NVCC_FLAGS) -DMATRIX_ELEM_DTYPE='$(MATRIX_ELEM_DTYPE)' -DADDITIONAL_OPTIM_FALG='$(ADDITIONAL_OPTIM_FALG)'
	@echo building BENCHMARK into: $@

##
### clean rules
##

clean:
	@rm $(BUILDDIR)/*.o $(TARGETDIR)/*
	@rm -r $(TARGETDIR)
	@echo directories cleaned

.PHONY: all debug clean gpu_build
