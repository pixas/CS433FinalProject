#include "tensor_core.h"
#include <cmath>
#include <cstring>

__half __float2half(const float &a) {
  uint32_t bits = *(uint32_t*)&a;
  int sign = (bits >> 31) & 1;
  int exponent = (bits >> 23) & 0xff;
  int mantissa = bits & 0x7fffff;
  __half result;
  if (exponent == 0) {
    // Denormalized number
    result.setX((unsigned short)((sign << 15) | (int)(mantissa / pow(2, -14))));
    return result;
  } else if (exponent == 0xff) {
    // Inf or NaN
    result.setX((unsigned short)((sign << 15) | (0x1f << 10) | (mantissa ? 0x200 : 0)));
    return result;
  } else {
    // Normalized number
    exponent = exponent - 127 + 15;
    if (exponent >= 0x1f) {
      // Overflow
      exponent = 0x1f;
      mantissa = 0;
    } else if (exponent <= 0) {
      // Underflow
      exponent = mantissa = 0;
    } else {
      mantissa = mantissa >> 13;
    }
    result.setX((unsigned short)((sign << 15) | (exponent << 10) | mantissa));
    return result;
  }
}

float __half2float(const __half &a) {
  uint16_t bits = (uint16_t)(a.getX());
  int sign = (bits >> 15) & 1;
  int exponent = (bits >> 10) & 0x1f;
  int mantissa = bits & 0x3ff;
  if (exponent == 0) {
    // Denormalized number
    return (float)(pow(-1, sign) * pow(2, -14) * (mantissa / pow(2, 10)));
  } else if (exponent == 0x1f) {
    // Inf or NaN
    return (float)((exponent == 0x1f && mantissa == 0) ? pow(-1, sign) * INFINITY : NAN);
  } else {
    // Normalized number
    return (float)(pow(-1, sign) * pow(2, exponent - 15) * (1 + mantissa / pow(2, 10)));
  }
}

float operator*(const __half &lh, const __half &rh) {
  uint16_t bits_a = lh.getX();
  uint16_t bits_b = rh.getX();
  int sign_a = (bits_a >> 15) & 1;
  int exponent_a = (bits_a >> 10) & 0x1f;
  int mantissa_a = bits_a & 0x3ff;
  int sign_b = (bits_b >> 15) & 1;
  int exponent_b = (bits_b >> 10) & 0x1f;
  int mantissa_b = bits_b & 0x3ff;
  int sign = sign_a ^ sign_b;
  int exponent = exponent_a + exponent_b - 15;
  int mantissa = mantissa_a * mantissa_b;
  if (exponent_a == 0) {
    // a is denormalized
    exponent_a -= 14;
    mantissa_a <<= 10;
  } else {
    mantissa_a += 0x400;
  }
  if (exponent_b == 0) {
    // b is denormalized
    exponent_b -= 14;
    mantissa_b <<= 10;
  } else {
    mantissa_b += 0x400;
  }
  exponent += exponent_a + exponent_b - 15;
  if (exponent > 30) {
    // Overflow
    return INFINITY;
  } else if (exponent <= 0) {
    // Underflow
    return 0.0;
  } else {
    mantissa = (mantissa_a * mantissa_b + 0x200) >> 10;
    if (mantissa & 0x400) {
      // Rounding
      mantissa += 0x400;
      exponent += 1;
    }
    if (exponent >= 0x1f) {
      // Overflow
      return INFINITY;
    } else {
      return (float)((sign << 15) | (exponent << 10) | (mantissa & 0x3ff));
    }
  }
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

// void GPU::SIM_LDG_INSTR() {
//   // LDG implementation
// }
void GPU::SIM_LDG_INSTR(bool E, unsigned sz, unsigned Rd, unsigned Ra, unsigned IMM) {
  // for: warp execution
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // LDG implementation
    unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned address = ra_data + IMM;
    unsigned data;
    if (E) {
      // 64-bit address
      uint64_t *ptr = (uint64_t *)&memory_[address / 4];
      data = *ptr;
    } else {
      // 32-bit address
      uint32_t *ptr = (uint32_t *)&memory_[address / 4];
      data = *ptr;
    }
    switch (sz) {
      case 1:
        // 8-bit data
        data &= 0xff;
        break;
      case 2:
        // 16-bit data
        data &= 0xffff;
        break;
      case 3:
        // 32-bit data
        data &= 0xffffffff;
        break;
      case 4:
        // 64-bit data
        data &= 0xffffffffffffffff;
        break;
    }
    unsigned &rd_data = regfile_[Rd * WARP_SIZE_ + threadIdx];
    rd_data = data;
  }
}
void GPU::SIM_STG_INSTR(unsigned Ra, unsigned Sb, bool E, unsigned imm, unsigned sz) {
  // STG implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; ++threadIdx) {
    unsigned & ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned & sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    
    unsigned data = sb_data;
    switch (sz) {
      case 1:
        // 8-bit data
        data &= 0xff;
        break;
      case 2:
        // 16-bit data
        data &= 0xffff;
        break;
      case 3:
        // 32-bit data
        data &= 0xffffffff;
        break;
      case 4:
        // 64-bit data
        data &= 0xffffffffffffffff;
        break;
    }
    if (E) {
      uint64_t * ptr = (uint64_t *)&memory_[(ra_data + imm) / 4];
      *ptr = data;
    } else {
      uint32_t * ptr = (uint32_t *)&memory_[(ra_data + imm) / 4];
      *ptr = data;
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

void GPU::SIM_S2R_INSTR(unsigned Rd, unsigned SR) {
  // S2R implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; ++threadIdx) {
    unsigned data = 0;
    unsigned & sr_data = regfile_[SR * WARP_SIZE_ + threadIdx];
    switch (sr_data) {
    case SR_LAINID:
      // SR_LANEID: thread index of the warp
      data = threadIdx;
      break;
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

void GPU::SIM_LOP3_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc,
                         unsigned imm) {
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


void GPU::SIM_SHF_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, bool dir, bool maxshift, bool HI) {
  // SHF implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    unsigned & ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned & sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
    unsigned & sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    unsigned val = (sc_data << 32) | ra_data;
    unsigned shift = HI ? (sb_data + 32) : sb_data;
    unsigned data;
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
        data = (int)val >> shift;
      }
      else {
        data = (int)val << shift;
      }
    }
    regfile_[Rd * WARP_SIZE_ + threadIdx] = (data & 0xffffffff);
  }
}

void GPU::SIM_CS2R_INSTR(unsigned Rd, unsigned SR) {
  // S2R implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; ++threadIdx) {
    uint64_t data = 0;
    regfile_[Rd * WARP_SIZE_ + threadIdx] = data;

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

void simMemcpy(void *dst, void *src, size_t count, enum cudaMemcpyKind kind,
               GPU &volta) {
  // sim cudaMemcpy
  // memcpy host memory to class GPU memory or
  // memcpy class GPU memory to host memory
  // Check that the destination and source pointers are within the bounds of the GPU memory
  if ((kind == MemcpyHostToDevice && (unsigned char*)dst < (unsigned  char*)(volta.memory_ + volta.memory_size_)) ||
      (kind == MemcpyDeviceToHost && (unsigned char*)src < (unsigned  char*)(volta.memory_ + volta.memory_size_))) {
    // Invalid destination or source pointer, throw an exception or return an error code
    throw std::invalid_argument("Invalid memory address");
  }

  // Copy memory between the host and the GPU
  unsigned char *dst_ptr = (unsigned char*)dst;
  unsigned char *src_ptr = (unsigned char*)src;
  if (kind == MemcpyHostToDevice) {
    // Copy from host to device
    memcpy(dst_ptr, src_ptr, count);
  } else {
    // Copy from device to host
    memcpy(dst_ptr, src_ptr, count);
  }
}

void wmma_kernel(__half *a, __half *b, float *c, float *d, dim3 &gridDim,
                 dim3 &blockDim, GPU &volta) {
  // device kernel function
  // gridDim & blockDim
  // assume c[0x0][0x28]=0
  const int c_0_28 = 0;
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
