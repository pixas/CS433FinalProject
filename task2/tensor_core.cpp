#include "tensor_core.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdlib.h>

using namespace std;

__half __float2half(const float &a) {
  uint32_t bits = *(uint32_t*)&a;
  int sign = (bits >> 31) & 0x1;
  int exponent = (bits >> 23) & 0xff;
  int mantissa = bits & 0x007fffff;
  __half result;
  if (exponent == 0) {
    // Denormalized number --> Underflow
    exponent = mantissa = 0;
  } else if (exponent == 0xff) {
    // Inf or NaN
    exponent = 0x1f;
    mantissa = mantissa ? 0x200 : 0;
  } else {
    // Normalized number
    if (exponent - 127 > 15) {
      // Overflow
      exponent = 0x1f;
      mantissa = 0;
    } else if (exponent - 127 >= -14) {
      // Normalized float -> normalized half, true exp in range [-14, 15], exp in range [113, 142]
      // exp - 127 + 15 = exp - 112
      exponent = exponent - 112;
      mantissa = (mantissa >> 13) & 0x3ff;
    } else if (exponent - 127 >= -24) {
      // Normalized float -> denormalized half, true exp in range [-24, -15], exp in range [103, 112]
      // exp - 103 = exp - 127 + 24, 126 - exp = -exp - 1 + 127
      mantissa = ((1 << (exponent - 103)) + (mantissa >> (126 - exponent))) & 0x3ff;
      exponent = 0;
    } else {
      // Underflow
      exponent = mantissa = 0;
    }
  }
  result.setX((unsigned short)((sign << 15) | (exponent << 10) | mantissa));
  return result;
}

void print_binary(uint32_t bits) {
  char binary[33];
  for (int i = 0; i < 32; ++i) {
    binary[31 - i] = (bits & (1 << i)) ? '1' : '0';
  }
  binary[32] = '\0';
  printf("%s\n", binary);
}

float __half2float(const __half &a) {
  uint16_t bits = (uint16_t)(a.getX());
  int sign = (bits >> 15) & 1;
  int exponent = (bits >> 10) & 0x1f;
  int mantissa = bits & 0x3ff;
  int result;
  if (exponent == 0) {
    // Denormalized number
    // return (float)(pow(-1, sign) * pow(2, -14) * (mantissa / pow(2, 10)));
    int highest_bit = 0;
    int temp = mantissa;
    while (temp) {
      ++highest_bit;
      temp >>= 1;
    }
    // exponent = - 14 - (11 - highest_bit) + 127;
    exponent = 102 + highest_bit;
    mantissa = ((1 << (exponent - 103)) + (mantissa >> (126 - exponent))) & 0x7fffff;
  } else if (exponent == 0x1f) {
    // Inf or NaN
    // return (float)((exponent == 0x1f && mantissa == 0) ? pow(-1, sign) * INFINITY : NAN);
    exponent = 0xff;
    mantissa = mantissa ? 0x400000 : 0x0;
  } else {
    // Normalized number
    // return (float)(pow(-1, sign) * pow(2, exponent - 15) * (1 + mantissa / pow(2, 10)));
    // exp_f - 127 = exp_h - 15, exp_f = exp_h + 112
    exponent = exponent + 112;
    mantissa = mantissa << 13;
  }
  result = (sign << 31) | (exponent << 23) | mantissa;
  return *reinterpret_cast<float*>(&result);
}

float operator*(const __half &lh, const __half &rh) {
  float a = __half2float(lh);
  float b = __half2float(rh);
  return a * b;
}

GPU::GPU() {
  // Initialize GPU resources reasonably, including regfile size and global
  // memory size, assuming sm=1 and warp_num=1
  constexpr unsigned MEMORY_SIZE = 1024 * 1024 * 1024;  // 1 GB
  memory_ = new unsigned[MEMORY_SIZE];
  memory_size_ = MEMORY_SIZE;
  // Initialize the register file and pre-register file
  unsigned REGFILE_SIZE = WARP_SIZE_ * 255;
  regfile_ = new unsigned[REGFILE_SIZE];
  unsigned PREGFILE_SIZE = WARP_SIZE_ * 7;
  pregfile_ = new bool[PREGFILE_SIZE];
  allocated_size_ = 0;
}

uint64_t concat(unsigned high, unsigned low) {
  uint64_t result = high;
  result = (result << 32) | low;
  return result;
}

/**
 * @brief Load data from global memory to register file
 * @param E: 1 if the address is 64-bit (load from Ra - Ra+1), 0 if the address is 32-bit (load from Ra)
 * @param sz: 64 means load 64-bit data, 128 means load 128-bit data
 * @param Rd: the first destination register (sz = 64 means load to Rd to Rd + 1 registers, sz = 128 means load to Rd to Rd + 3 registers)
 * @param Ra: the first source register containing the base address
 * @param imm: the immediate value that contains the offset
 */
void GPU::SIM_LDG_INSTR(bool E, unsigned sz, unsigned Rd, unsigned Ra, unsigned imm) {
  // for: warp execution
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // LDG implementation
    unsigned *data_ptr;
    if (E) {
      unsigned &ra_data_0 = regfile_[Ra * WARP_SIZE_ + threadIdx];
      unsigned &ra_data_1 = regfile_[(Ra + 1) * WARP_SIZE_ + threadIdx];
      uint64_t addr = concat(ra_data_1, ra_data_0) + (uint64_t)imm;
      data_ptr = &memory_[addr / 4];  // unsigned is 4 bytes
    } else {
      unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
      unsigned addr = ra_data + imm;
      data_ptr = &memory_[addr / 4];  // unsigned is 4 bytes
    }
    switch (sz) {
      case 64:
        // 64-bit data
        regfile_[Rd * WARP_SIZE_ + threadIdx] = *data_ptr;
        regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = *(data_ptr + 1);
        break;
      case 128:
        // 128-bit data
        regfile_[Rd * WARP_SIZE_ + threadIdx] = *data_ptr;
        regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = *(data_ptr + 1);
        regfile_[(Rd + 2) * WARP_SIZE_ + threadIdx] = *(data_ptr + 2);
        regfile_[(Rd + 3) * WARP_SIZE_ + threadIdx] = *(data_ptr + 3);
        break;
    }
  }
}

/**
 * @brief Store data from register file to global memory
 * @param E: 1 if the address is 64-bit (load from Ra - Ra+1), 0 if the address is 32-bit (load from Ra)
 * @param sz: 64 means load 64-bit data, 128 means load 128-bit data
 * @param Sb: the first source register (sz = 64 means store data from Rd to Rd + 1 registers, sz = 128 means store data from Rd to Rd + 3 registers)
 * @param Ra: the first source register containing the base address
 * @param imm: the immediate value that contains the offset
 */
void GPU::SIM_STG_INSTR(bool E, unsigned sz, unsigned Sb, unsigned Ra, unsigned imm) {
  // STG implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; ++threadIdx) {
    unsigned * data_ptr;
    if (E) {
      unsigned &ra_data_0 = regfile_[Ra * WARP_SIZE_ + threadIdx];
      unsigned &ra_data_1 = regfile_[(Ra + 1) * WARP_SIZE_ + threadIdx];
      uint64_t addr = concat(ra_data_1, ra_data_0) + (uint64_t)imm;
      data_ptr = &memory_[addr / 4];
    } else {
      unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
      unsigned addr = ra_data + imm;
      data_ptr = &memory_[addr / 4];
    }

    switch (sz) {
      case 64:
        *(data_ptr) = regfile_[Sb * WARP_SIZE_ + threadIdx];
        *(data_ptr + 1) = regfile_[(Sb + 1) * WARP_SIZE_ + threadIdx];
        break;
      case 128:
        *(data_ptr) = regfile_[Sb * WARP_SIZE_ + threadIdx];
        *(data_ptr + 1) = regfile_[(Sb + 1) * WARP_SIZE_ + threadIdx];
        *(data_ptr + 2) = regfile_[(Sb + 2) * WARP_SIZE_ + threadIdx];
        *(data_ptr + 3) = regfile_[(Sb + 3) * WARP_SIZE_ + threadIdx];
        break;
    }
  }
}
void GPU::SIM_HMMA_INSTR_STEP0() {
  // HMMA.STEP0 implementation
}
void GPU::SIM_HMMA_INSTR_STEP1() {
  // HMMA.STEP1 implementation
}
void GPU::SIM_HMMA_INSTR_STEP2() {
  // HMMA.STEP2 implementation
}
void GPU::SIM_HMMA_INSTR_STEP3() {
  // HMMA.STEP3 implementation
}

/**
 * @brief S2R instruction
 * @param Rd: the destination register
 * @param SR: store data type
 */
void GPU::SIM_S2R_INSTR(unsigned Rd, s_reg_t SR) {
  // S2R implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; ++threadIdx) {
    unsigned data = 0;
    switch (SR) {
    case SR_LAINID:
      // SR_LANEID: thread index of the warp
      data = threadIdx;
      break;
    // The four following cases are not used in this project, so the implementation is fake
    case SR_TID_X:
      // SR_TID.X: threadIdx.x
      data = SR_TID_X;
      break;
    case SR_TID_Y:
      // SR_TID.Y: threadIdx.y
      data = SR_TID_Y;
      break;
    case SR_CTAID_X:
      // SR_CTAID.X: blockIdx.x
      data = SR_CTAID_X;
      break;
    case SR_CTAID_Y:
      // SR_CTAID.Y: blockIdx.y
      data = SR_CTAID_Y;
      break;
    default:
      // Invalid SR value
      break;
    }
    regfile_[Rd * WARP_SIZE_ + threadIdx] = data;
  }
}

void GPU::SIM_IMAD_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, bool wide, unsigned fmt) {
  // IMAD implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    unsigned & ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned & sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    if (wide) {
      if (fmt) {
        int64_t data;
        // TODO: int x int -> 64bit? reg file order?
        data = ((int)ra_data * (int)sb_data) & 0xffffffffffffffff;
        int64_t * sc_data = (int64_t * )&regfile_[Sc * WARP_SIZE_ + threadIdx];
        int64_t * rd_data = (int64_t *)&regfile_[Rd * WARP_SIZE_ + threadIdx];  
        *rd_data = data + (*sc_data);
      } else {
        uint64_t data;
        data = (ra_data * sb_data) & 0xffffffffffffffff;
        uint64_t * sc_data = (uint64_t*)&regfile_[Sc * WARP_SIZE_ + threadIdx];
        uint64_t * rd_data = (uint64_t *)&regfile_[Rd * WARP_SIZE_ + threadIdx];
        *rd_data = data + (*sc_data);
      }
    } else {
      if (fmt) {
        int data;
        data = ((int)ra_data * (int)sb_data) & 0xffffffffffffffff;
        int * sc_data = (int * )&regfile_[Sc * WARP_SIZE_ + threadIdx];
        int * rd_data = (int *)&regfile_[Rd * WARP_SIZE_ + threadIdx];  
        *rd_data = data + (*sc_data);
      } else {
        unsigned data;
        data = (ra_data * sb_data) & 0xffffffffffffffff;
        unsigned * sc_data = (unsigned*)&regfile_[Sc * WARP_SIZE_ + threadIdx];
        unsigned * rd_data = (unsigned *)&regfile_[Rd * WARP_SIZE_ + threadIdx];
        *rd_data = data + (*sc_data);
      }

    }
  }
}

void GPU::SIM_LOP3_INSTR(
  unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc,
  unsigned imm
) {
  // for: warp execuation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // LOP3 implementation
    unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned &sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    unsigned &sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
    unsigned data = 0;
    if (imm & 0x01) data |= (~ra_data) & (~sb_data) & (~sc_data);
    if (imm & 0x02) data |= (~ra_data) & (~sb_data) & (sc_data);
    if (imm & 0x04) data |= (~ra_data) & (sb_data) & (~sc_data);
    if (imm & 0x08) data |= (~ra_data) & (sb_data) & (sc_data);
    if (imm & 0x10) data |= (ra_data) & (~sb_data) & (~sc_data);
    if (imm & 0x20) data |= (ra_data) & (~sb_data) & (sc_data);
    if (imm & 0x40) data |= (ra_data) & (sb_data) & (~sc_data);
    if (imm & 0x80) data |= (ra_data) & (sb_data) & (sc_data);
    regfile_[Rd * WARP_SIZE_ + threadIdx] = data;
  }
}

/**
 * @brief SHF instruction
 * @param Rd: the destination register
 * @param Ra: the register containing the low 32 bits of the value to be shifed
 * @param Sb: the register containing the shift bits
 * @param Sc: the register containing the high 32 bits of the value to be shifed
 * @param dir: the direction of the shift (0: left, 1: right)
 * @param maxshift: logical shift (1) or arithmetic shift (0)
 * @param HI: shift bits plus 32 (1) or not (0)
 */
void GPU::SIM_SHF_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, bool dir, bool maxshift, bool HI) {
  // SHF implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    unsigned & ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned & sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
    unsigned & sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    uint64_t val = concat(sc_data, ra_data);
    unsigned shift = HI ? (sb_data + 32) : sb_data;
    uint64_t data;
    if (maxshift) {
      // logical shift
      if (dir) {
        // right shift
        data = val >> shift;
      }
      else {
        data = val << shift;
      }
    } else {
      // arithmetic shift
      if (dir) {
        data = (int64_t)val >> shift;
      }
      else {
        data = (int64_t)val << shift;
      }
    }
    regfile_[Rd * WARP_SIZE_ + threadIdx] = (data & 0xffffffff);
  }
}

/**
 * @brief CS2R instruction
 * @param Rd: the destination register
 * @param SR: store data type (SRZ)
 */
void GPU::SIM_CS2R_INSTR(unsigned Rd, s_reg_t SR) {
  // S2R implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; ++threadIdx) {
    if (SR != SRZ)
      printf("CS2R error: SR != SRZ\n");
    // 64-bit 0
    regfile_[Rd * WARP_SIZE_ + threadIdx] = 0;
    regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = 0;
  }
}

void GPU::SIM_LEA_INSTR(bool HI, bool X, unsigned Rd, unsigned Ra, unsigned Sb,
                        unsigned imm, unsigned Pd0, unsigned Ps0) {
  // for: warp execuation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // LEA implementation
    unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned &sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    uint64_t data = ra_data;
    if (HI)
      data = data >> (32 - imm);
    else
      data = data << imm;
    data += sb_data;
    if (X) data += pregfile_[Ps0 * WARP_SIZE_ + threadIdx];
    if (Pd0 != 7)
      pregfile_[Pd0 * WARP_SIZE_ + threadIdx] = ((data >> 32) & 0x1);
    data &= 0xffffffff;
    regfile_[Rd * WARP_SIZE_ + threadIdx] = data;
  }
}

void GPU::SIM_EXIT_INSTR() {
  // EXIT implementation

  // Free any dynamically allocated memory
  delete[] memory_;
  delete[] regfile_;
  delete[] pregfile_;
  allocated_size_ = 0;
  memory_size_ = 0;
  // Close any open files or connections
  // (if applicable)

  // Perform any other necessary shutdown tasks
  // (if applicable)
}

void simMalloc(void **ptr, size_t size, GPU &volta) {
  // sim cudaMalloc
  // Request GPU memory
  if (volta.allocated_size_ + size > volta.memory_size_) {
    // Not enough memory available, throw an exception or return an error code
    throw std::bad_alloc();
  }

  // Allocate memory on the GPU
  *ptr = volta.memory_ + volta.allocated_size_;
  volta.allocated_size_ += size;
}

inline int check_in_GPU(void *ptr, size_t count, GPU &volta) {
  // Check that the whole memory block is within the bounds of the GPU memory
  if (
    (unsigned char*)ptr >= (unsigned char*)(volta.memory_) &&
    (unsigned char*)ptr + count <= (unsigned char*)(volta.memory_ + volta.memory_size_)
  )
    return 1;
  return 0;
}

void simMemcpy(void *dst, void *src, size_t count, enum cudaMemcpyKind kind,
               GPU &volta) {
  // sim cudaMemcpy
  // memcpy host memory to class GPU memory or
  // memcpy class GPU memory to host memory
  // Check that the destination and source pointers are within the bounds of the GPU memory
  if (
    (kind == MemcpyHostToDevice && check_in_GPU(dst, count, volta)) ||
    (kind == MemcpyDeviceToHost && check_in_GPU(src, count, volta))
  ) {
    // Invalid destination or source pointer, throw an exception or return an error code
    throw std::invalid_argument("Invalid memory address");
  }

  // Copy memory between the host and the GPU
  memcpy(dst, src, count);
}

void wmma_kernel(__half *a, __half *b, float *c, float *d, dim3 &gridDim,
                 dim3 &blockDim, GPU &volta) {
  // device kernel function
  // gridDim & blockDim
  // assume c[0x0][0x28]=0
  const int c_0_28 = 0;
  // 16 x 16 mma

  // load a, b, c to gpu memory
  // thread group 0, 2 load a[0:3], 
  // thread group 4, 6 load a[4:7]
  // thread group 1, 3 load a[8:11]
  // thread group 5, 7 load a[12:15]
  // volta.SIM_LDG_INSTR(1, 128, 2, ((unsigned int)a / 4), 0);
  // volta.SIM_LDG_INSTR(1, 128)
  // volta.SIM_IMAD_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, unsigned fmt, bool wide);  // SASS: IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
  // add instruction you need,sim_imad_instr() is just an example
}

void gemm(float *a, float *b, float *c, float *d) {
  // host function gemm
  // mxnxk=? According to the pytorch source code, find the matrix size of the
  // conv2d operator of Resnet18 when doing matrix multiplication after
  // im2col. Remember to turn off cudnn and use the cublas library We need to
  // considerthe settings of blockDim and gridDim, we limit only one warp, how
  // to slicethe matrix to a matrix of 16*16*16 size, what to do if it cannot
  // be divisible evenly
}

void im2col() {
  // ref caffe
}
void conv() {}

int main() {
//   float a_list[16] = {pow(2, -20), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5};
//   float b_list[16] = {0, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.7, 0.8, -0.9, 1, 1.1, 1.2, -1.3, 1.4, 1.5};
//   float c_list[16];

//   __half ha_list[16], hb_list[16];

//   for (int i = 0; i < 16; i++) {
//     ha_list[i] = __float2half(a_list[i]);
//     hb_list[i] = __float2half(b_list[i]);
//     std::cout << a_list[i] << " " << __half2float(ha_list[i]) << std::endl;
//   }

//   for (int i = 0; i < 16; i++) {
//     c_list[i] = a_list[i] * b_list[i];
//     float hc = ha_list[i] * hb_list[i];
//     printf("%f %f %f %f\n", c_list[i], hc, c_list[i] - hc, __half2float(__float2half(c_list[i])));
//   }

  // unsigned a = 0x3f800000;
  // unsigned b = 0x10000001;
  // uint64_t c = concat(a, b);
  // printf("%x\n%x\n%lx\n", a, b, c);
  // printf("%x\n%x\n", (uint64_t)a, (uint64_t)b);

  GPU volta;
  // volta.SIM_LDG_INSTR(1, 64, 0, 0, 0);
  // volta.SIM_STG_INSTR(1, 64, 0, 0, 0);
  // volta.SIM_SHF_INSTR(0, 1, 2, 3, 0, 0, 0);
}
