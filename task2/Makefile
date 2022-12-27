include buildenv.mk

INFER_OBJ := $(TARGET_DIR)/inference
SRC_ACT_DIR := src/activation/
SRC_CONV_DIR := src/conv/
SRC_UTILS_DIR := src/utils/
INFER_SRCS := $(wildcard $(SRC_ACT_DIR)*.cu) \
			 $(wildcard $(SRC_CONV_DIR)*.cu) \
			 $(wildcard $(SRC_UTILS_DIR)*.cu) \
			 $(wildcard $(SRC_UTILS_DIR)*.cpp) \
			 src/inference.cu

ORACLE_OBJ := $(TARGET_DIR)/oracle
ORACLE_SRCS := src/oracle.cpp

all: $(INFER_OBJ) $(ORACLE_OBJ)

$(INFER_OBJ): $(INFER_SRCS)
	@$(NVCC) $(NVCC_FLAGS) $(OPENCV_INC) $(INFER_SRCS) -o $(INFER_OBJ) $(OPENCV_LIB)
	@echo "Inference executable is generated at $(INFER_OBJ)"


$(ORACLE_OBJ): $(ORACLE_SRCS)
	@$(CXX) $(CXX_FLAGS) $(OPENCV_INC) $(ORACLE_SRCS) -o $(ORACLE_OBJ) $(OPENCV_LIB)
	@echo "Oracle executable is generated at $(ORACLE_OBJ)"

.PHONEY: clean
clean:
	@$(RM) $(TARGET_DIR)/*
	@echo "All executables at $(TARGET_DIR) are removed"
